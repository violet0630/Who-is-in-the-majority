"""Validate frozen supplemental inputs; execute exactly one new trajectory on request.

Default mode is offline and does not import the API client. Existing run directories
are rejected, including interrupted runs: recovery needs a separate reviewed step.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
from random import Random
import sys


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    # Git normalizes repository text to LF. Preserve content checks across Windows
    # working trees and fresh clones without treating CRLF as a scientific change.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Input path escapes its root")
    return path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(repo, prepared):
    manifest = read(prepared / "preparation_manifest.json")
    for relative, expected in manifest.items():
        require(sha(contained(prepared, relative)) == expected, f"Prepared input changed: {relative}")
    sources = read(prepared / "source_hashes.json")
    for relative, expected in sources.items():
        require(sha(contained(repo, relative)) == expected, f"Source changed: {relative}")
    sys.path.insert(0, str(repo))
    from majority.artifacts import load_bank, load_network
    from majority.preparation import draw_profiles, seed_diagnostics
    protocol = read(prepared / "protocol.json")
    require(protocol.get("hash_policy") == "SHA256 of text bytes with CRLF normalized to LF", "Unknown hash policy")
    initialization = read(repo / "configs/initialized_20261116.json")
    require(protocol["settings"] == initialization["settings"], "Settings differ from original initialization")
    require((protocol["size"], protocol["k"], protocol["observers"]) == (1000, 200, 50), "Population changed")
    tasks = read(prepared / "tasks.json")
    expected_ids = {f"g{g}-r{r}-{b}" for g in range(3) for r in range(3, 10) for b in ("TH", "TL", "U0")}
    require(len(tasks) == 63 and {t["task_id"] for t in tasks} == expected_ids, "Task coverage mismatch")
    require(sum(t["phase"] == "technical_trial" for t in tasks) == 9, "Trial count mismatch")
    groups, networks = {}, {}
    for task in tasks:
        g, r, b = task["graph"], task["repeat"], task["branch"]
        require(task["task_id"] == f"g{g}-r{r}-{b}", "Task identity mismatch")
        require(task["group_seed"] == 18000 + g * 10 + r, "Process seed mismatch")
        require(task["network"] == f"experiments/2026-11-16/03_positions/network-{g}.json", "Network mismatch")
        require(task["phase"] == ("technical_trial" if r == 3 else "remaining"), "Trial assignment mismatch")
        path = contained(prepared, task["prepared"])
        require(sha(path) == task["prepared_sha256"], "Position input changed")
        if (g, r) in groups:
            require(groups[g, r]["path"] == path, "Branches have different profiles")
            continue
        data = read(path)
        require((data["graph"], data["repeat"]) == (g, r), "Group identity mismatch")
        profiles = draw_profiles(1000, Random(16000 + g * 10 + r))
        require(data["profiles"] == [asdict(p) for p in profiles], "Profile seed reconstruction failed")
        if g not in networks:
            networks[g] = load_network(repo / task["network"])
        network = networks[g]
        require(len(network.following) == 1000 and len(network.observers) == 50, "Network size mismatch")
        quotas, resources, ig = [], [], {}
        for branch in ("TH", "TL", "U0"):
            nodes = data["positions"][branch]
            selected = frozenset(nodes)
            require(len(nodes) == len(selected) == 200 and all(0 <= i < 1000 for i in nodes), "Invalid advocate count")
            require(not selected & network.observers, "Observer selected as advocate")
            quotas.append(Counter((network.communities[i], profiles[i].strength) for i in nodes))
            diag = asdict(seed_diagnostics(network, profiles, selected))
            ig[branch] = diag["illusion"]
            resources.append([sum(len(network.fans[i]) for i in nodes),
                              sum(profiles[i].baseline for i in nodes),
                              sum(6 * profiles[i].baseline / (1 + 5 * profiles[i].baseline) for i in nodes)])
        require(quotas[0] == quotas[1] == quotas[2], "Community/strength quota mismatch")
        for i, j in ((0, 1), (0, 2), (1, 2)):
            require(all(2 * abs(x-y)/(x+y) <= .05 + 1e-12 for x, y in zip(resources[i], resources[j])), "Resource matching failed")
        require(ig["TH"] > ig["TL"], "Position contrast failed; review without silently redrawing")
        groups[g, r] = dict(path=path, data=data, profiles=profiles)
    _, bank = load_bank(repo / "experiments/2026-11-16/01_texts_50/bank.json")
    return protocol, tasks, groups, networks, bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--task", help="Exactly one task, e.g. g0-r3-TH")
    parser.add_argument("--runs", type=Path, help="New output root; never use the original archive")
    parser.add_argument("--execute", action="store_true", help="Make paid API calls for --task; omit for offline validation")
    args = parser.parse_args()
    repo, prepared = args.repo.resolve(), args.prepared.resolve()
    protocol, tasks, groups, networks, bank = validate(repo, prepared)
    if args.task:
        require(args.task in {t["task_id"] for t in tasks}, "Unknown task")
    report = dict(status="offline_checks_passed", groups=len(groups), trajectories=len(tasks),
                  technical_trial_tasks=[t["task_id"] for t in tasks if t["phase"] == "technical_trial"],
                  openai_package_present=importlib.util.find_spec("openai") is not None,
                  api_calls_made=0, credential_validation="not_performed")
    print(json.dumps(report), flush=True)
    if not args.execute:
        return
    require(args.task is not None and args.runs is not None, "Execution requires --task and --runs")
    root = args.runs.resolve()
    archive = (repo / "experiments/2026-11-16").resolve()
    require(not root.is_relative_to(archive) and not root.is_relative_to(prepared), "Output overlaps frozen artifacts")
    task = next(t for t in tasks if t["task_id"] == args.task)
    destination = root / task["task_id"]
    require(not destination.exists(), "Run directory already exists; no automatic overwrite or resume")
    # Import credentials only at explicit execution time; never print or save keys.
    from scripts.majority_study import run_one
    from majority.llm import canonical_model, load_settings
    config, _ = load_settings(repo / ".env")
    require(canonical_model(config.model) == "deepseek-flash", "Model differs from original experiment")
    require(config.max_tokens == 4096, "Output limit differs from original experiment")
    runtime = dict(base_url=config.base_url, model=config.model, max_tokens=config.max_tokens,
                   timeout=180, thinking="disabled", temperature=0,
                   prepared_manifest_sha256=sha(prepared / "preparation_manifest.json"),
                   runner_sha256=sha(Path(__file__)))
    root.mkdir(parents=True, exist_ok=True)
    runtime_path = root / "runtime.json"
    if runtime_path.exists():
        require(read(runtime_path) == runtime, "Batch runtime changed; review before mixing results")
    else:
        with runtime_path.open("x", encoding="utf-8") as stream:
            json.dump(runtime, stream, indent=2)
    destination.mkdir(exist_ok=False)
    with (destination / "input_identity.json").open("x", encoding="utf-8") as stream:
        json.dump(task, stream, indent=2)
    group = groups[task["graph"], task["repeat"]]
    result = run_one(networks[task["graph"]], group["profiles"],
                     frozenset(group["data"]["positions"][task["branch"]]), bank,
                     protocol["settings"], task["group_seed"], task["branch"], destination, True)
    require(len(result["metrics"]) == 10 and result["tail_rounds"] == 3, "Incomplete trajectory")
    require((destination / "trajectory.jsonl.gz").exists(), "Missing trajectory")
    print(json.dumps(dict(status="trajectory_finished_pending_independent_audit", task=args.task,
                          summary=result["summary"])), flush=True)


if __name__ == "__main__":
    main()
