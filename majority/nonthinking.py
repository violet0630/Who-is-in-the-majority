"""Inspect visible response metadata; stop a shared batch on a policy violation."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from threading import Event, Lock


class NonThinkingError(RuntimeError):
    """Not a transient transport/schema error: never automatically retry."""


def response_anomaly(response, limit):
    for choice in response.choices:
        message = choice.message.model_dump()
        for field in ("reasoning_content", "reasoning", "reasoning_details", "thinking"):
            value = message.get(field)
            if value and (not isinstance(value, str) or value.strip()):
                return f"{field}_present"
        if re.match(r"\s*(?:```[^\n]*\n\s*)?<(?:think|thinking|analysis)>",
                    message.get("content") or "", flags=re.IGNORECASE):
            return "thinking_embedded_in_content"
    usage = response.usage.model_dump() if response.usage else {}
    details = usage.get("completion_tokens_details") or {}
    if (details.get("reasoning_tokens") or 0) > 0 or (usage.get("reasoning_tokens") or 0) > 0:
        return "positive_reasoning_tokens"
    if (usage.get("completion_tokens") or 0) > limit:
        return "output_above_requested_limit"
    return None


class NonThinkingGuard:
    """One instance per batch, shared across task and observer worker threads."""

    def __init__(self, stop_file=None):
        self.stopped = Event()
        self.lock = Lock()
        self.reason = ""
        self.stop_file = Path(stop_file) if stop_file is not None else None
        if self.stop_file is not None and self.stop_file.exists():
            self.reason = "unresolved_nonthinking_stop_file"
            self.stopped.set()

    def check(self):
        if self.stopped.is_set():
            raise NonThinkingError(self.reason)

    def inspect(self, response, limit):
        anomaly = response_anomaly(response, limit)
        if anomaly:
            with self.lock:
                if not self.stopped.is_set():
                    self.reason = anomaly
                    self.stopped.set()
                    if self.stop_file is not None:
                        # Persist before another run can silently reset the circuit.
                        with self.stop_file.open("x", encoding="utf-8") as stream:
                            json.dump({"reason": anomaly, "response_id": response.id,
                                       "response_model": response.model,
                                       "time": datetime.now(timezone.utc).isoformat()}, stream)
            raise NonThinkingError(anomaly)
        # Another in-flight request may already have stopped the batch.
        self.check()
        usage = response.usage.model_dump() if response.usage else {}
        return {"status": "no_visible_anomaly", "usage_reported": response.usage is not None,
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "completion_tokens": usage.get("completion_tokens")}
