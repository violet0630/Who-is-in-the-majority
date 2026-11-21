"""Recompute the main experiment 2 Q and J metrics and audit model calls."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, task, message):
    if not condition:
        raise ValueError(f"{task}: {message}")


def verify_seal(directory, bundle_id, task, runtime):
    identity = {"bundle_id": bundle_id, "task": task, "runtime": runtime}
    require(read(directory / "identity.json") == identity, task["id"], "run identity differs")
    seal = read(directory / "complete.json")
    require(seal["bundle_id"] == bundle_id and seal["task_id"] == task["id"]
            and set(seal["files"]) == {"result.json", "trajectory.jsonl.gz", "calls.jsonl"},
            task["id"], "invalid completion seal")
    for name, expected in seal["files"].items():
        with (directory / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        require(actual == expected, task["id"], f"completion checksum differs for {name}")


def replay(directory, task, observers, advocates, expected_llm):
    metrics = read(directory / "result.json")["metrics"]
    expected_events, messages, rounds = set(), {}, []
    with gzip.open(directory / "trajectory.jsonl.gz", "rt", encoding="utf-8") as stream:
        header = json.loads(next(stream))
        require(header["kind"] == "manifest", task, "missing trajectory manifest")
        setup = header["data"]
        require(setup["llm"] == expected_llm and setup["thinking"] == "disabled"
                and len(advocates) == len(setup["advocates"]) and set(setup["advocates"]) == advocates
                and set(setup["network"]["observers"]) == observers, task, "trajectory identity differs")
        for line in stream:
            row = json.loads(line)
            if row["kind"] != "round":
                continue
            data = row["data"]
            number = len(rounds) + 1
            require(data["round"] == number and number <= 10, task, "invalid round order")
            rounds.append(number)
            messages.update({msg["id"]: msg["stance"] for msg in data["published"]})
            nodes = {node["node"]: node for node in data["nodes"]}
            require(set(nodes) == set(range(1000)) and len(data["nodes"]) == 1000,
                    task, "invalid population")
            require(all(node["before"]["stance"] == node["after"]["stance"]
                        == (-1 if i in advocates else 1) for i, node in nodes.items()),
                    task, "private stances differ from the frozen allocation")
            q = j = 0
            for observer in observers:
                node = nodes[observer]
                reading = node["reading"]["read"]
                if reading:
                    expected_events.add((observer, number))
                q += node["after"]["perceived_support"] < 0.5
                j += sum(messages[mid] == -1 for mid in reading) > len(reading) / 2
            metric = metrics[number - 1]
            require(metric["round"] == number and abs(metric["perceived_majority"] - q / 50) < 1e-10
                    and abs(metric["feed_majority"] - j / 50) < 1e-10, task, f"Q or J mismatch in round {number}")
    require(len(rounds) == len(metrics) == 10, task, "incomplete trajectory")
    return expected_events


def audit_calls(directory, expected_events, model, task):
    successful_events, responses, states = set(), Counter(), Counter()
    with (directory / "calls.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            call = record["data"]
            context = call["context"]
            event = (context["node"], context["round"])
            require(record["kind"] == "llm_call" and event in expected_events
                    and context["run_id"] == directory.name and call["model"] == model,
                    task, "unexpected observer, model or task in call log")
            require(call["temperature"] == 0 and context["thinking"] == "disabled",
                    task, "perception configuration changed")
            states[call["status"]] += 1
            if call["status"] == "ok":
                successful_events.add(event)
                responses[call["response"]["model"]] += 1
    require(successful_events == expected_events, task, "incomplete observer updates")
    return {"observer_updates": len(successful_events), "responses": dict(responses), "statuses": dict(states)}


def audit(prepared, pilot):
    manifest = read(prepared / "manifest.json")
    runtime = read(prepared / "runtime.json")
    tasks = [t for t in manifest["tasks"] if t["pilot"] or not pilot]
    rows = {}
    for task in tasks:
        identifier = task["id"]
        run = prepared / "runs" / f"{identifier}-{manifest['bundle_id'][:12]}"
        verify_seal(run, manifest["bundle_id"], task, runtime)
        observers = set(read(prepared / task["network_file"])["data"]["network"]["observers"])
        advocates = set(read(prepared / task["positions_file"])["positions"]["TH"])
        require(len(observers) == 50 and len(advocates) == task["k"] and not observers & advocates,
                identifier, "invalid frozen observers or advocates")
        expected = replay(run, identifier, observers, advocates, runtime["llm"])
        rows[identifier] = audit_calls(run, expected, runtime["llm"]["model"], identifier)
    return {"scope": "pilot" if pilot else "full", "runs_audited": len(rows), "runs": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.prepared.resolve(), args.pilot), ensure_ascii=False))


if __name__ == "__main__":
    main()
