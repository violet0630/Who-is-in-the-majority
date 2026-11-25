"""Explicitly authorized transport-only continuation of an immutable v2 bundle.

The original runtime/identity denotes the initial execution stage. Per-task
continuation receipts record the actual new transport and audit byte boundary.
Model/input/source checks remain enabled; completed files are never rewritten.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path

from majority.llm import load_settings
from scripts.main2_v2 import audit_bundle
from scripts.majority_factorial import (PROJECT, completed_result, digest, execute_task,
    file_hash, load_prepared, read_json, run_directory, write_json)
from scripts.summarize_factorial import summarize
from scripts.gateway_nonthinking import Circuit, GuardedClient, TASK_DIRECTORY, quarantine_cache


def validate_transport_switch(original, current, approved_url):
    """Only this explicitly supplied route may differ; all other settings match."""
    if current["llm"]["base_url"] != approved_url:
        raise ValueError("Current endpoint differs from the explicitly approved endpoint")
    comparable = {**current, "llm": {**current["llm"], "base_url": original["llm"]["base_url"]}}
    if comparable != original:
        raise ValueError("Only base_url may change; model, generation settings and packages must match")


def current_runtime():
    config, _ = load_settings(PROJECT / ".env")
    return {"llm": {**asdict(config), "timeout": 180}, "temperature": 0, "thinking": "disabled",
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "openai", "python-dotenv")}}


def continue_runs(prepared, approved_url, *, allow_api=False):
    if not allow_api:
        raise ValueError("Explicit --allow-api is required")
    prepared = Path(prepared).resolve()
    manifest = load_prepared(prepared)  # Keep original source and all input checks.
    original = read_json(prepared / "runtime.json")
    current = current_runtime()
    validate_transport_switch(original, current, approved_url)
    import scripts.majority_study as study
    from majority.llm import open_client as original_open_client
    circuit = Circuit()
    def guarded_open_client(config, key):
        return GuardedClient(original_open_client(config, key), TASK_DIRECTORY.get(), circuit)
    study.open_client = guarded_open_client
    pending, preserved = [], {}
    for task in manifest["tasks"]:
        directory = run_directory(prepared, manifest, task)
        if (directory / "complete.json").exists():
            completed_result(prepared, manifest, task, original)
            preserved[task["id"]] = file_hash(directory / "complete.json")
        else:
            pending.append(task)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    adapter_sources = {}
    for path in (Path(__file__), PROJECT / "scripts/gateway_nonthinking.py"):
        relative = f"transport-sources/{stamp}/{path.name}"
        destination = prepared / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(path.read_bytes())
        adapter_sources[relative] = file_hash(destination)
    receipt = {"created_at": datetime.now(timezone.utc).isoformat(), "bundle_id": manifest["bundle_id"],
               "authorization": "User explicitly requested switching to api.1nvictus.me and continuing; user asserts same underlying model",
               "original_runtime": original, "continuation_runtime": current,
               "original_runtime_sha256": file_hash(prepared / "runtime.json"),
               "adapter_source_sha256": file_hash(__file__), "preserved_complete_seals": preserved,
               "adapter_sources": adapter_sources,
               "explicit_nonthinking": True, "request_addition": {"reasoning_effort": "none"},
               "guard": "Stop on returned reasoning, output above requested limit, or 3 consecutive API errors",
               "pending_tasks": [t["id"] for t in pending],
               "identity_semantics": "Original runtime in identity.json is the initial stage; actual continuation route is recorded here and in task receipts"}
    receipt_path = prepared / f"gateway-transition-{stamp}.json"
    write_json(receipt_path, receipt)

    def execute(task):
        # Recheck before each task; never change the frozen scientific implementation.
        load_prepared(prepared)
        validate_transport_switch(original, current_runtime(), approved_url)
        directory = run_directory(prepared, manifest, task)
        circuit.check()
        reset = quarantine_cache(directory)
        token = TASK_DIRECTORY.set(directory)
        calls = directory / "calls.jsonl"
        boundary = {"transition_file": receipt_path.name, "transition_sha256": file_hash(receipt_path),
                    "task_id": task["id"], "actual_runtime": current,
                    "reasoning_effort": "none", "cache_reset": reset,
                    "previous_calls_bytes": calls.stat().st_size if calls.exists() else 0,
                    "previous_calls_sha256": file_hash(calls) if calls.exists() else None}
        directory.mkdir(parents=True, exist_ok=True)
        # Existing execute_task expects an identity even in an existing directory.
        if not (directory / "identity.json").exists():
            from scripts.majority_factorial import task_identity
            write_json(directory / "identity.json", task_identity(manifest, task, original))
        write_json(directory / f"gateway-continuation-{stamp}.json", boundary)
        print(f"RESUME {task['id']}", flush=True)
        try:
            execute_task(prepared, manifest, task, original)
        except Exception:
            circuit.stopped.set()
            raise
        finally:
            TASK_DIRECTORY.reset(token)
        print(f"COMPLETED {task['id']}", flush=True)
        return task["id"]

    statuses, failed = [], False
    # Finish and independently audit one full recovered run before expanding.
    if pending:
        first = pending.pop(0)
        try:
            execute(first)
            from scripts.main2_v2 import audit_trajectory
            from scripts.majority_factorial import inside
            network = read_json(inside(prepared, first["network_file"]))["data"]["network"]
            advocates = set(read_json(inside(prepared, first["positions_file"]))["positions"]["TH"])
            trial = audit_trajectory(run_directory(prepared, manifest, first) / "trajectory.jsonl.gz",
                {**manifest["protocol"]["settings"], "rounds": 10, "rho": first["rho"], "alpha": .05},
                set(network["observers"]), advocates)
            write_json(prepared / f"nonthinking-trial-{stamp}.json", {"task": first["id"], "audit": trial})
            statuses.append({"task": first["id"], "status": "complete"})
            print("RECOVERY TRIAL AUDIT PASSED; continuing remaining tasks", flush=True)
        except Exception as error:
            circuit.stopped.set()
            write_json(prepared / f"gateway-batch-{stamp}.json", {
                "statuses": [{"task": first["id"], "status": "failed", "error_type": type(error).__name__}], "failed": True})
            raise
    iterator = iter(pending)
    with ThreadPoolExecutor(max_workers=2) as pool:
        active = {}
        for _ in range(2):
            task = next(iterator, None)
            if task:
                active[pool.submit(execute, task)] = task
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                task = active.pop(future)
                try:
                    future.result()
                    statuses.append({"task": task["id"], "status": "complete"})
                except Exception as error:
                    failed = True
                    statuses.append({"task": task["id"], "status": "failed", "error_type": type(error).__name__})
                    print(f"STOP DISPATCH: {task['id']} {type(error).__name__}", flush=True)
            if not failed:
                while len(active) < 2:
                    task = next(iterator, None)
                    if task is None:
                        break
                    active[pool.submit(execute, task)] = task
    write_json(prepared / f"gateway-batch-{stamp}.json", {"statuses": statuses, "failed": failed})
    for task in manifest["tasks"]:
        if task["id"] in preserved:
            directory = run_directory(prepared, manifest, task)
            if file_hash(directory / "complete.json") != preserved[task["id"]]:
                raise ValueError("A previously completed run changed")
            completed_result(prepared, manifest, task, original)
    if failed:
        raise RuntimeError("Gateway continuation failed; unstarted tasks left untouched")
    audit = audit_bundle(prepared)
    output = prepared / "full_analysis"
    if not output.exists():
        summarize(prepared, output)
    if not (output / "trajectory_audit.json").exists():
        write_json(output / "trajectory_audit.json", audit)
    transitions = sorted(p.name for p in prepared.glob("gateway-transition-*.json"))
    note = {"bundle_id": manifest["bundle_id"], "transitions": transitions,
            "model_equivalence": "User asserted; gateway response aliases do not independently prove model equivalence"}
    if not (output / "gateway_provenance.json").exists():
        write_json(output / "gateway_provenance.json", note)
        with (output / "report.md").open("a", encoding="utf-8") as stream:
            stream.write("\n## 接口切换记录\n\n本批实验经用户授权从DeepSeek直连接口切换至1nvictus网关继续执行。"
                         "用户确认底层模型相同；响应模型别名不能独立验证等价性。原始完成结果未重写，"
                         "未完成运行复用既有成功请求。具体切换任务、调用日志字节边界与实际配置见")
            stream.write("、".join(f"[{name}](../{name})" for name in transitions) + "。\n")
    from scripts.summarize_factorial import summarize_calls
    excluded = {}
    for task in manifest["tasks"]:
        directory = run_directory(prepared, manifest, task)
        marker = directory / "nonthinking_cache_reset.json"
        if not marker.exists():
            continue
        reset = read_json(marker)
        before = summarize_calls(directory / reset["backup_file"])
        trusted = summarize_calls(directory / reset["trusted_prefix_file"])
        excluded[task["id"]] = {key: before[key]-trusted[key] for key in before}
    if not (output / "quarantined_usage.json").exists():
        write_json(output / "quarantined_usage.json", excluded)
        with (output / "report.md").open("a", encoding="utf-8") as stream:
            stream.write("\n续跑已显式加入 reasoning_effort=none。早期网关阶段思考设置不明确的响应已备份隔离，"
                         "恢复时只保留最初直连阶段的可信缓存。每次新响应检查推理内容、推理用量和输出上限。"
                         "网关输出硬上限在小请求测试中未生效；本地检查仅能在响应返回后检测，不能阻止上游生成或计费。"
                         "上面的调用用量不含被隔离阶段的请求；额外消耗见[隔离阶段用量](quarantined_usage.json)。\n")
    print(f"COMPLETE: {audit['runs_audited']}/36 audited; report and four figures ready", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--approved-url", required=True)
    parser.add_argument("--allow-api", action="store_true")
    args = parser.parse_args()
    continue_runs(args.prepared, args.approved_url, allow_api=args.allow_api)
