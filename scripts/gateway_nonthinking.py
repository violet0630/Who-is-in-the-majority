"""Audited non-thinking transport and reversible cache quarantine for v2 recovery."""

from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace

from openai import APIConnectionError, APIStatusError
from scripts.majority_factorial import file_hash, read_json, write_json


TASK_DIRECTORY = ContextVar("gateway_task_directory")


class GatewayGuardError(RuntimeError):
    pass


class Circuit:
    def __init__(self):
        self.stopped = Event()
        self.lock = Lock()
        self.errors = 0

    def check(self):
        if self.stopped.is_set():
            raise GatewayGuardError("Gateway circuit stopped; no further requests dispatched")

    def success(self):
        with self.lock:
            self.errors = 0

    def failure(self):
        with self.lock:
            self.errors += 1
            if self.errors >= 3:
                self.stopped.set()


def response_anomaly(response, limit):
    for choice in response.choices:
        message = choice.message.model_dump()
        if (message.get("reasoning_content") or "").strip():
            return "reasoning_content_present"
        if (message.get("content") or "").lstrip().startswith("<think>"):
            return "thinking_embedded_in_content"
    usage = response.usage.model_dump() if response.usage else {}
    if ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0) > 0:
        return "positive_reasoning_tokens"
    if (usage.get("completion_tokens") or 0) > limit:
        return "output_above_requested_limit"
    return None


class GuardedClient:
    def __init__(self, client, directory, circuit):
        self.client, self.circuit = client, circuit
        self.log = (Path(directory) / "nonthinking_transport.jsonl").open("a", encoding="utf-8")
        self.lock = Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def record(self, row):
        with self.lock:
            self.log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            self.log.flush()

    def create(self, **kwargs):
        self.circuit.check()
        kwargs = {**kwargs, "reasoning_effort": "none"}
        if (kwargs.get("extra_body") or {}).get("thinking") != {"type": "disabled"}:
            self.circuit.stopped.set()
            raise GatewayGuardError("Frozen thinking-disabled field unexpectedly changed")
        row = {"time": datetime.now(timezone.utc).isoformat(), "reasoning_effort": "none",
               "thinking": {"type": "disabled"}, "max_tokens": kwargs["max_tokens"],
               "temperature": kwargs.get("temperature"), "model": kwargs["model"],
               "messages_sha256": hashlib.sha256(json.dumps(kwargs["messages"], sort_keys=True,
                                                             ensure_ascii=False).encode()).hexdigest()}
        try:
            response = self.client.chat.completions.create(**kwargs)
        except (APIConnectionError, APIStatusError) as error:
            self.circuit.failure()
            self.record({**row, "status": "api_error", "error_type": type(error).__name__,
                         "http_status": getattr(error, "status_code", None)})
            raise
        anomaly = response_anomaly(response, kwargs["max_tokens"])
        self.record({**row, "status": "rejected" if anomaly else "accepted_transport",
                     "anomaly": anomaly, "response_id": response.id, "response_model": response.model,
                     "reasoning_content_characters": sum(len((c.message.model_dump().get("reasoning_content") or ""))
                                                          for c in response.choices),
                     "usage": response.usage.model_dump() if response.usage else None})
        if anomaly:
            self.circuit.stopped.set()
            raise GatewayGuardError(anomaly)
        self.circuit.success()
        return response

    def close(self):
        try:
            self.client.close()
        finally:
            self.log.close()


def quarantine_cache(directory):
    """Restore only the prefix before the first gateway attempt, keeping full backup."""
    directory = Path(directory).resolve()
    if (directory / "complete.json").exists():
        raise ValueError("Never quarantine a completed run")
    if (directory / "result.json").exists():
        raise ValueError("Unsealed result requires inspection before cache reset")
    calls = directory / "calls.jsonl"
    marker = directory / "nonthinking_cache_reset.json"
    if marker.exists():
        record = read_json(marker)
        prefix_file = directory / record["trusted_prefix_file"]
        backup = directory / record["backup_file"]
        if file_hash(prefix_file) != record["trusted_prefix_sha256"] or file_hash(backup) != record["backup_sha256"]:
            raise ValueError("Quarantine provenance changed")
        prefix = prefix_file.read_bytes()
        if not calls.read_bytes().startswith(prefix):
            raise ValueError("Live cache lost trusted original prefix")
        return record
    receipts = sorted(directory.glob("gateway-continuation-*.json"))
    if not calls.exists():
        directory.mkdir(parents=True, exist_ok=True)
        calls.write_bytes(b"")
    first = read_json(receipts[0]) if receipts else {
        "previous_calls_bytes": calls.stat().st_size, "previous_calls_sha256": file_hash(calls)}
    qdir = directory / "quarantine" / "before-explicit-nonthinking"
    if not qdir.resolve().is_relative_to(directory):
        raise ValueError("Quarantine must stay inside the task directory")
    qdir.mkdir(parents=True, exist_ok=True)
    backup = qdir / "calls-before-reset.jsonl"
    data = backup.read_bytes() if backup.exists() else calls.read_bytes()
    prefix = data[:first["previous_calls_bytes"]]
    expected = first["previous_calls_sha256"]
    if len(prefix) != first["previous_calls_bytes"] or (expected is not None and hashlib.sha256(prefix).hexdigest() != expected):
        raise ValueError("Original direct-API cache prefix hash mismatch")
    for line in prefix.decode("utf-8").splitlines():
        json.loads(line)
    live = calls.read_bytes()
    if backup.exists() and live not in (data, prefix):
        raise ValueError("Interrupted quarantine has unexpected new calls")
    if not backup.exists():
        with backup.open("xb") as stream:
            stream.write(data)
    trusted = qdir / "trusted-original-prefix.jsonl"
    if not trusted.exists():
        with trusted.open("xb") as stream:
            stream.write(prefix)
    elif trusted.read_bytes() != prefix:
        raise ValueError("Existing trusted prefix differs")
    staged = qdir / "reset.tmp"
    staged.write_bytes(prefix)
    os.replace(staged, calls)
    record = {"first_gateway_receipt": receipts[0].name if receipts else None,
              "backup_file": backup.relative_to(directory).as_posix(), "backup_sha256": file_hash(backup),
              "trusted_prefix_file": trusted.relative_to(directory).as_posix(),
              "trusted_prefix_sha256": file_hash(trusted), "trusted_prefix_bytes": len(prefix),
              "quarantined_bytes": len(data)-len(prefix),
              "reason": "User approved isolating all uncertain gateway-stage responses; restore only original direct-API prefix"}
    write_json(marker, record)
    return record
