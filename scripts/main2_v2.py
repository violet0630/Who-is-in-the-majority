"""Authorized v2 workflow: nine pilots, independent trajectory audit, remaining 27, plots."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from majority.models import message_order
from scripts.majority_factorial import (PROJECT, completed_result, inside, load_prepared,
    prepare, read_json, run_batch, run_directory, write_json)
from scripts.summarize_factorial import summarize


def audit_trajectory(path, expected_config, observers, advocates, expected_rounds=10):
    """Independent checks of actual trajectories, not only aggregate result fields."""
    messages, round_count, multiple_nodes = {}, 0, 0
    previous = {}
    expected_metrics = None
    manifest_seen = False
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            data = entry["data"]
            if entry["kind"] == "manifest":
                for key, value in expected_config.items():
                    if data["config"][key] != value:
                        raise ValueError(f"Actual run config mismatch: {key}")
                manifest_seen = True
            if entry["kind"] == "metrics":
                if expected_metrics is None or any(abs(data[key] - value) > 1e-12 for key, value in expected_metrics.items()):
                    raise ValueError("Public-expression or majority metrics do not match trajectory")
            if entry["kind"] != "round":
                continue
            round_count += 1
            if data["round"] != round_count:
                raise ValueError("Invalid round sequence")
            counts = Counter(m["author"] for m in data["published"])
            multiple_nodes += sum(n > 1 for n in counts.values())
            targets, notifications = {}, Counter()
            latest_public = {}
            for m in sorted(data["published"], key=lambda x: message_order(SimpleNamespace(**x))):
                if m["id"] in messages or m["round"] != round_count or counts[m["author"]] > 3:
                    raise ValueError("Invalid message identity, time or count")
                messages[m["id"]] = m
                latest_public[m["author"]] = m["stance"]
                if m["author"] in advocates and m["stance"] != -1:
                    raise ValueError("Advocate public stance changed")
                if m["original_id"] is not None:
                    target = messages[m["original_id"]]
                    seen = targets.setdefault(m["author"], set())
                    if m["original_id"] in seen or target["author"] == m["author"] or target["action"] != "post":
                        raise ValueError("Invalid/repeated comment target")
                    seen.add(m["original_id"])
                    notifications[target["author"]] += 1
                elif counts[m["author"]] != 1 or m["stance"] != (-1 if m["author"] in advocates else 1):
                    raise ValueError("Original post violates private stance or action gate")
            if len(data["nodes"]) != 1000 or {r["node"] for r in data["nodes"]} != set(range(1000)):
                raise ValueError("Node records incomplete")
            for row in data["nodes"]:
                node, reading = row["node"], row["reading"]
                before, after = row["before"], row["after"]
                if before["stance"] != (-1 if node in advocates else 1) or after["stance"] != before["stance"]:
                    raise ValueError("Private stance changed")
                if node in previous and before != previous[node]:
                    raise ValueError("State update did not carry forward synchronously")
                if node not in previous and (before["perceived_support"] != .5 or before["feedback_value"] != 0):
                    raise ValueError("Unexpected initial state")
                read, visible = reading["read"], reading["visible"]
                if len(read) != min(11, len(visible)) or len(set(read)) != len(read) or not set(read) <= set(visible):
                    raise ValueError("Reading sample rule violated")
                for mid in visible:
                    if node not in messages[mid]["audience"] or not round_count - 1 <= messages[mid]["round"] <= round_count:
                        raise ValueError("Nonlocal/stale visible message")
                if len(row["notifications"]) != notifications[node]:
                    raise ValueError("Comment notifications lost or duplicated")
                if not read:
                    if row["perception"] is not None or after["perceived_support"] != before["perceived_support"]:
                        raise ValueError("Empty reading did not carry forward")
                elif node not in observers:
                    authors = {}
                    for mid in sorted(read, key=lambda mid: message_order(SimpleNamespace(**messages[mid]))):
                        authors[messages[mid]["author"]] = messages[mid]["stance"]
                    expected = sum(s == 1 for s in authors.values()) / len(authors)
                    if abs(after["perceived_support"] - expected) > 1e-12 or row["perception"]["model"] != "local_proxy":
                        raise ValueError("Local author-deduplicated perception differs")
                elif row["perception"]["model"] == "local_proxy":
                    raise ValueError("Observer did not use LLM")
                previous[node] = after
            observed = [r for r in data["nodes"] if r["node"] in observers]
            expected_metrics = {
                "public_support": sum(latest_public.get(i) == 1 for i in observers) / len(observers),
                "public_advocate_stance": sum(latest_public.get(i) == -1 for i in observers) / len(observers),
                "public_conformity": sum(latest_public.get(i) == -1 for i in observers) / len(observers),
                "opponent_silence": sum(i not in latest_public for i in observers) / len(observers),
                "perceived_majority": sum(r["after"]["perceived_support"] < .5 for r in observed) / len(observers),
                "feed_majority": sum(sum(messages[mid]["stance"] == -1 for mid in r["reading"]["read"])
                                     > len(r["reading"]["read"]) / 2 for r in observed) / len(observers)}
            if sum(len(r["notifications"]) for r in data["nodes"]) != sum(m["original_id"] is not None for m in data["published"]):
                raise ValueError("Notification total mismatch")
        if not manifest_seen or round_count != expected_rounds:
            raise ValueError("Missing manifest or rounds")
    return {"rounds": round_count, "multi_comment_node_rounds": multiple_nodes, "messages": len(messages)}


def audit_bundle(prepared, pilot=False):
    manifest = load_prepared(prepared)
    runtime = read_json(prepared / "runtime.json")
    results = {}
    for task in manifest["tasks"]:
        if pilot and not task["pilot"]:
            continue
        completed_result(prepared, manifest, task, runtime)
        network = read_json(inside(prepared, task["network_file"]))["data"]["network"]
        advocates = set(read_json(inside(prepared, task["positions_file"]))["positions"]["TH"])
        results[task["id"]] = audit_trajectory(
            run_directory(prepared, manifest, task) / "trajectory.jsonl.gz",
            {**manifest["protocol"]["settings"], "rounds": 10, "rho": task["rho"], "alpha": .05},
            set(network["observers"]), advocates)
    return {"bundle_id": manifest["bundle_id"], "runs_audited": len(results), "runs": results}


def run_workflow(prepared):
    prepared = Path(prepared).resolve()
    for batch, pilot, folder in (("pilot", True, "pilot_analysis"), ("remaining", False, "full_analysis")):
        run_batch(prepared, batch, allow_api=True)
        audit = audit_bundle(prepared, pilot=pilot)
        output = prepared / folder
        if not output.exists():
            summarize(prepared, output, pilot=pilot)
        audit_path = output / "trajectory_audit.json"
        if not audit_path.exists():
            write_json(audit_path, audit)
        print(f"AUDIT PASSED: {audit['runs_audited']} complete runs; {folder}", flush=True)
    print("COMPLETE: 36 runs, audited trajectories, report and four figures", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--allow-api", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(PROJECT / "configs/main2_factorial_v2.json", args.prepared)
    if args.allow_api:
        run_workflow(args.prepared)


if __name__ == "__main__":
    main()
