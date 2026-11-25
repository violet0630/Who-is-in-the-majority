"""Main experiment 2: explicit offline preparation and separately gated API runs.

Future commands (do not execute until authorized):
  python -m scripts.majority_factorial prepare --output runs/main2_new
  python -m scripts.majority_factorial run --prepared runs/main2_new --batch pilot --allow-api

Importing this module does not search positions, write files, or contact an API.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

from majority.artifacts import load_bank, load_network
from majority.models import Profile


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT / "configs/main2_factorial.json"
PRIMARY_COMPARISONS = ["count_endpoints", "activity_endpoints", "interaction_endpoints"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(root, relative):
    """Only resolve paths contained in the specified bundle/project."""
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Path must stay inside {root}: {relative}")
    return path


def source_hashes():
    paths = list((PROJECT / "majority").glob("*.py"))
    paths += list((PROJECT / "majority/prompts").glob("*.md"))
    paths += [PROJECT / "configs/study.json", PROJECT / "scripts/__init__.py",
              PROJECT / "scripts/majority_study.py", Path(__file__),
              PROJECT / "scripts/summarize_factorial.py", PROJECT / "scripts/main2_v2.py"]
    return {path.relative_to(PROJECT).as_posix(): file_hash(path) for path in sorted(set(paths))}


def validate_protocol(protocol, baseline):
    if protocol.get("schema_version") not in (1, 2) or protocol.get("position_strategy") != "TH":
        raise ValueError("Only schema 1 or 2 / TH is supported")
    for field in ("graphs", "repeats", "k_grid", "rho_grid"):
        values = protocol[field]
        if not values or len(set(values)) != len(values) or values != sorted(values):
            raise ValueError(f"{field} must be nonempty, unique, and sorted")
    expected_repeats = [0] if protocol["schema_version"] == 2 else [0, 1, 2]
    if protocol["graphs"] != [0, 1, 2] or protocol["repeats"] != expected_repeats:
        raise ValueError("Unexpected archived graph/profile groups")
    if protocol["schema_version"] == 2:
        expected = {"read_limit": 11, "window": 2, "max_comments": 3,
                    "comment_alpha": 1.0, "comment_beta": 9.0, "cross_fan_comments": True}
        if (protocol.get("settings") != expected or not protocol.get("reuse_prepared")
                or protocol["analysis"].get("mode") != "three_graph_descriptive"):
            raise ValueError("V2 requires frozen local-perception rules and reused positions")
    n = baseline["size"]
    if n != 1000 or baseline["observer_count"] != 50:
        raise ValueError("Expected the frozen 1000-node / 50-observer baseline")
    if any(type(k) is not int or not 0 < k < n / 2 for k in protocol["k_grid"]):
        raise ValueError("k must be a positive integer minority")
    if len(protocol["k_grid"]) < 2 or len(protocol["rho_grid"]) < 2:
        raise ValueError("Endpoint contrasts require at least two levels of both factors")
    if any(type(rho) not in (int, float) or not math.isfinite(rho) or rho <= 0
           for rho in protocol["rho_grid"]):
        raise ValueError("rho must be positive and finite")
    search = protocol["search"]
    if search["reference_k"] != baseline["k"] or search["reference_k"] not in protocol["k_grid"]:
        raise ValueError("Reference k must reuse the archived baseline")
    if search["reference_rho"] != baseline["settings"]["rho"]:
        raise ValueError("Freeze the same search reference rho as experiment 1")
    if any(type(search[key]) is not int or search[key] < 1 for key in ("steps", "restarts")):
        raise ValueError("Search budgets must be positive integers")
    if search["tolerance"] != 0.05 or not search["seed_namespace"]:
        raise ValueError("Preserve the 5% matching protocol and specify a seed namespace")
    pilot = protocol["pilot"]
    if (pilot["graph"] not in protocol["graphs"] or pilot["repeat"] not in protocol["repeats"]
            or not pilot["k_grid"] or len(set(pilot["k_grid"])) != len(pilot["k_grid"])
            or not set(pilot["k_grid"]) <= set(protocol["k_grid"])):
        raise ValueError("Pilot must be a fixed subset of the planned conditions")
    execution = protocol["execution"]
    if (execution["gateway_limit"] != 16 or
            any(type(execution[key]) is not int or execution[key] < 1
                for key in ("run_workers", "llm_workers"))):
        raise ValueError("Positive worker counts and the inherited 16-call gateway limit are required")
    settings = baseline["settings"]
    if (baseline["alpha"] != 0.05 or baseline["beta"] != 10
            or settings["rounds"] != 10 or settings["tail_rounds"] != 3):
        raise ValueError("The inherited runner requires alpha=.05, beta=10, 10 rounds and tail=3")
    analysis = protocol["analysis"]
    if (analysis["primary_comparisons"] != PRIMARY_COMPARISONS
            or analysis["bootstrap_repeats"] < 100
            or not 0 < analysis["confidence_level"] < 1
            or not 0 < analysis["test_alpha"] < 1
            or not 0 < analysis["coverage_target"] <= 1):
        raise ValueError("Invalid analysis protocol")


def search_seed(namespace, graph, repeat, k):
    return int(digest([namespace, graph, repeat, k]), 16)


def relative_gap(a, b):
    return 2 * abs(a - b) / (a + b) if a + b else 0.0


def seed_diagnostics(network, profiles, seeds, rhos):
    """Independent edge-based audit; deliveries are counts, not unique readers."""
    n = len(profiles)
    nodes = sorted(seeds)
    fans = {u: {i for i, followed in enumerate(network.following) if u in followed} for u in nodes}
    audience = set().union(*fans.values())
    ordinary = set(range(n)) - set(nodes)
    illusion = [i for i in sorted(ordinary)
                if 2 * len(network.following[i].intersection(nodes)) > len(network.following[i])]
    exposure = []
    for rho in rhos:
        g = {u: rho * profiles[u].baseline / (1 + (rho - 1) * profiles[u].baseline) for u in nodes}
        exposure.append({"rho": rho, "expected_speakers": sum(g.values()),
                         "expected_deliveries": sum(g[u] * len(fans[u]) for u in nodes),
                         "expected_observer_deliveries": sum(g[u] * len(fans[u] & network.observers)
                                                             for u in nodes)})
    total = sum(len(fans[u]) for u in nodes)
    return {"illusion": len(illusion) / len(ordinary), "illusion_nodes": illusion,
            "observer_illusion": len(set(illusion) & network.observers) / len(network.observers),
            "fan_total": total, "fan_union": len(audience), "duplicate_fan_links": total - len(audience),
            "external_fan_union": len(audience - set(nodes)),
            "observer_fan_union": len(audience & network.observers),
            "baseline_total": sum(profiles[u].baseline for u in nodes),
            "strong_count": sum(profiles[u].strength == 0.8 for u in nodes), "exposure": exposure}


def audit_positions(network, profiles, positions, k, search, rhos):
    if set(positions) != {"TH", "TL", "U0"} or len(profiles) != len(network.following):
        raise ValueError("Expected all three source positions and a profile for each node")
    labels = sorted(set(network.communities))
    available = [sum(c == label and i not in network.observers
                     for i, c in enumerate(network.communities)) for label in labels]
    total = sum(available)
    quotas = [k * count // total for count in available]
    for i in sorted(range(len(labels)), key=lambda j: (-(k * available[j] % total), j))[:k - sum(quotas)]:
        quotas[i] += 1
    cells = {}
    for branch, nodes in positions.items():
        if (len(nodes) != k or len(set(nodes)) != k or not set(nodes) <= set(range(len(profiles)))
                or set(nodes) & network.observers):
            raise ValueError(f"Invalid or observer seeds: {branch}")
        if [sum(network.communities[i] == label for i in nodes) for label in labels] != quotas:
            raise ValueError(f"Community quota mismatch: {branch}")
        cells[branch] = Counter((network.communities[i], profiles[i].strength) for i in nodes)
    if any(value != cells["U0"] for value in cells.values()):
        raise ValueError("Community/strength composition differs across source positions")
    all_rhos = sorted(set(rhos) | {search["reference_rho"]})
    diagnostics = {branch: seed_diagnostics(network, profiles, nodes, all_rhos)
                   for branch, nodes in positions.items()}
    matching = {}
    for a, b in (("TH", "TL"), ("TH", "U0"), ("TL", "U0")):
        first, second = diagnostics[a], diagnostics[b]
        gaps = {field: relative_gap(first[field], second[field]) for field in ("fan_total", "baseline_total")}
        speakers = [next(row["expected_speakers"] for row in item["exposure"]
                         if row["rho"] == search["reference_rho"]) for item in (first, second)]
        gaps["initial_speech_total_at_reference_rho"] = relative_gap(*speakers)
        if any(gap > search["tolerance"] + 1e-12 for gap in gaps.values()):
            raise ValueError(f"Source position matching failed: {a}-{b}: {gaps}")
        matching[f"{a}-{b}"] = gaps
    warnings = []
    if diagnostics["TH"]["illusion"] <= diagnostics["TL"]["illusion"]:
        warnings.append("TH does not exceed TL in strict structural IG; review before API execution")
    if diagnostics["TH"]["illusion"] <= diagnostics["U0"]["illusion"]:
        warnings.append("TH does not exceed the random reference in strict IG")
    return {"diagnostics": diagnostics, "matching": matching, "warnings": warnings}


def build_tasks(protocol, groups):
    tasks = []
    for group_index, group in enumerate(groups):
        conditions = [(k, rho) for k in protocol["k_grid"] for rho in protocol["rho_grid"]]
        shift = group_index % len(conditions)
        for k, rho in conditions[shift:] + conditions[:shift]:
            graph, repeat = group["graph"], group["repeat"]
            identifier = f"g{graph}-r{repeat}-k{k}-rho{rho:g}"
            tasks.append({"id": identifier, "graph": graph, "repeat": repeat, "k": k, "rho": rho,
                          "group_seed": 18000 + graph * 10 + repeat,
                          "network_file": group["network_file"], "profiles_file": group["profiles_file"],
                          "positions_file": group["positions"][str(k)],
                          "pilot": graph == protocol["pilot"]["graph"] and repeat == protocol["pilot"]["repeat"]
                          and k in protocol["pilot"]["k_grid"]})
    if len({task["id"] for task in tasks}) != len(tasks):
        raise ValueError("Duplicate task identity")
    return tasks


def prepare(protocol_path, output):
    """Explicitly invoked offline stage; no credentials or API client are loaded."""
    protocol = read_json(protocol_path)
    baseline = read_json(inside(PROJECT, protocol["baseline"]))
    validate_protocol(protocol, baseline)
    reused_root = inside(PROJECT, protocol["reuse_prepared"]) if protocol.get("reuse_prepared") else None
    reused = load_prepared(reused_root, check_code=False) if reused_root else None
    output = Path(output).resolve()
    archive = inside(PROJECT, protocol["archive"])
    if output.is_relative_to(archive):
        raise ValueError("New preparation must stay outside the experiment-1 archive")
    output.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    files = {}

    def freeze(source, relative):
        destination = inside(output, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(Path(source).read_bytes())
        files[relative] = file_hash(destination)
        return relative

    def save(relative, data):
        write_json(inside(output, relative), data)
        files[relative] = file_hash(inside(output, relative))
        return relative

    save("protocol.json", protocol)
    save("inputs/baseline.json", baseline)
    bank_file = freeze(inside(PROJECT, baseline["text_bank"]), "inputs/bank.json")
    bank_data = read_json(inside(output, bank_file))["data"]
    # Baseline bank hashes use spaced JSON rather than this module's canonical digest.
    bank_hash = hashlib.sha256(json.dumps(bank_data["bank"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if bank_hash != baseline["text_bank_sha256"] or bank_data["study"] != read_json(PROJECT / "configs/study.json"):
        raise ValueError("Frozen text bank or study differs from the baseline")
    if bank_data["study"]["advocate_stance"] != -1:
        raise ValueError("Main experiment 2 preserves opposing advocates")
    for relative in sources:
        freeze(inside(PROJECT, relative), f"source/{relative}")
    groups = []
    # Freeze every graph/profile input before any position search.
    for graph in protocol["graphs"]:
        network_file = freeze(archive / f"03_positions/network-{graph}.json", f"inputs/network-{graph}.json")
        network = load_network(inside(output, network_file))
        if len(network.communities) != baseline["size"] or len(network.observers) != baseline["observer_count"]:
            raise ValueError("Archived network size/observer count mismatch")
        for repeat in protocol["repeats"]:
            source_file = freeze(archive / f"03_positions/g{graph}-r{repeat}_positions.json",
                                 f"inputs/g{graph}-r{repeat}-source.json")
            original = read_json(inside(output, source_file))
            if original["graph"] != graph or original["repeat"] != repeat:
                raise ValueError("Archived group identity mismatch")
            profiles_file = save(f"inputs/g{graph}-r{repeat}-profiles.json", original["profiles"])
            groups.append({"graph": graph, "repeat": repeat, "network_file": network_file,
                           "source_file": source_file, "profiles_file": profiles_file, "positions": {}})
    from scripts.majority_study import select_positions
    audits = []
    for group in groups:
        network = load_network(inside(output, group["network_file"]))
        profiles = tuple(Profile(**row) for row in read_json(inside(output, group["profiles_file"])))
        for k in protocol["k_grid"]:
            seed = search_seed(protocol["search"]["seed_namespace"], group["graph"], group["repeat"], k)
            if reused is not None:
                old_group = next(row for row in reused["groups"]
                                 if (row["graph"], row["repeat"]) == (group["graph"], group["repeat"]))
                for field in ("network_file", "profiles_file"):
                    if file_hash(inside(output, group[field])) != file_hash(inside(reused_root, old_group[field])):
                        raise ValueError(f"Reused {field} differs from archived group")
                old_file = old_group["positions"][str(k)]
                freeze(inside(reused_root, old_file), f"provenance/g{group['graph']}-r{group['repeat']}-k{k}.json")
                positions = read_json(inside(reused_root, old_file))["positions"]
                origin = "reused_main2_positions"
            elif k == protocol["search"]["reference_k"]:
                positions = read_json(inside(output, group["source_file"]))["positions"]
                origin = "reused_archived_positions"
            else:
                try:
                    selected, _ = select_positions(network, profiles, k, seed,
                        steps=protocol["search"]["steps"], restarts=protocol["search"]["restarts"],
                        rho=protocol["search"]["reference_rho"])
                except Exception as error:
                    save(f"failures/g{group['graph']}-r{group['repeat']}-k{k}-search.json",
                         {"stage": "search", "graph": group["graph"], "repeat": group["repeat"],
                          "k": k, "seed": seed, "error_type": type(error).__name__})
                    raise
                positions = {branch: sorted(nodes) for branch, nodes in selected.items()}
                origin = "new_structural_search"
            candidate = {"graph": group["graph"], "repeat": group["repeat"], "k": k,
                         "origin": origin, "search_seed": seed if origin == "new_structural_search" else None,
                         "positions": positions}
            save(f"inputs/g{group['graph']}-r{group['repeat']}-k{k}-candidate.json", candidate)
            try:
                audit = audit_positions(network, profiles, positions, k, protocol["search"], protocol["rho_grid"])
            except Exception as error:
                save(f"failures/g{group['graph']}-r{group['repeat']}-k{k}-audit.json",
                     {**candidate, "stage": "audit", "error_type": type(error).__name__, "error": str(error)})
                raise
            entry = {"graph": group["graph"], "repeat": group["repeat"], "k": k, "origin": origin,
                     "search_seed": seed if origin == "new_structural_search" else None,
                     "positions": positions, "audit": audit}
            relative = f"inputs/g{group['graph']}-r{group['repeat']}-k{k}-positions.json"
            group["positions"][str(k)] = save(relative, entry)
            audits.append(entry)
            print(f"Prepared g{group['graph']}-r{group['repeat']} k={k}; IG={audit['diagnostics']['TH']['illusion']:.4f}", flush=True)
    tasks = build_tasks(protocol, groups)
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "protocol": protocol, "baseline_file": "inputs/baseline.json", "bank_file": bank_file,
                "files": files, "source_hashes": sources, "groups": groups, "tasks": tasks}
    manifest["bundle_id"] = digest(manifest)
    write_json(output / "manifest.json", manifest)
    lines = ["# 主实验二离线准备报告", "", f"任务数：{len(tasks)}；仅准备输入，没有调用API。", "",
             "同一组、同一k的所有rho共享TH名单；k=200复用主实验一名单。", "",
             "| 图/组 | k | TH IG | TL IG | 随机 IG | 观测者 IG | 提示 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in audits:
        d = row["audit"]["diagnostics"]
        lines.append(f"| g{row['graph']}-r{row['repeat']} | {row['k']} | {d['TH']['illusion']:.2%} | "
                     f"{d['TL']['illusion']:.2%} | {d['U0']['illusion']:.2%} | {d['TH']['observer_illusion']:.2%} | "
                     + ("；".join(row["audit"]["warnings"]) or "原约束通过") + " |")
    lines += ["", "初始传播诊断保存在各positions JSON中。投递量不是去重人数或实际阅读量，不能跨k/rho强行匹配。",
              "本报告没有评定结构差异是否足够。必须审阅后再批准API运行。", ""]
    (output / "preparation_report.md").write_text("\n".join(lines), encoding="utf-8")
    return manifest


def load_prepared(prepared, *, check_code=True):
    prepared = Path(prepared).resolve()
    manifest = read_json(prepared / "manifest.json")
    unsigned = {key: value for key, value in manifest.items() if key != "bundle_id"}
    if digest(unsigned) != manifest["bundle_id"]:
        raise ValueError("Preparation manifest changed")
    for relative, expected in manifest["files"].items():
        if file_hash(inside(prepared, relative)) != expected:
            raise ValueError(f"Frozen input changed: {relative}")
    if check_code and source_hashes() != manifest["source_hashes"]:
        raise ValueError("Code/prompt/study changed since preparation; do not mix protocols")
    baseline = read_json(inside(prepared, manifest["baseline_file"]))
    validate_protocol(manifest["protocol"], baseline)
    expected_tasks = build_tasks(manifest["protocol"], manifest["groups"])
    if manifest["tasks"] != expected_tasks:
        raise ValueError("Task grid does not match the frozen protocol")
    return manifest


def task_settings(baseline, protocol, task):
    return {**baseline["settings"], **protocol.get("settings", {}),
            "fraction": task["k"] / baseline["size"], "rho": task["rho"],
            "llm_workers": protocol["execution"]["llm_workers"]}


def run_directory(prepared, manifest, task):
    return Path(prepared) / "runs" / f"{task['id']}-{manifest['bundle_id'][:12]}"


def task_identity(manifest, task, runtime):
    return {"bundle_id": manifest["bundle_id"], "task": task, "runtime": runtime}


def validate_result(result, task, baseline, settings):
    if (result.get("branch") != "TH" or result.get("size") != baseline["size"]
            or result.get("k") != task["k"] or result.get("seed") != task["group_seed"]
            or result.get("settings") != settings or result.get("tail_rounds") != settings["tail_rounds"]):
        raise ValueError("Result belongs to a different condition/configuration")
    metrics = result["metrics"]
    if [row["round"] for row in metrics] != list(range(1, settings["rounds"] + 1)):
        raise ValueError("Incomplete/duplicate round sequence")
    for row in metrics:
        if not math.isclose(row["true_advocate_support"], task["k"] / baseline["size"], abs_tol=1e-12):
            raise ValueError("Private advocate fraction changed")
        if not all(type(row[field]) in (int, float) and math.isfinite(row[field]) and 0 <= row[field] <= 1
                   for field in ("perceived_majority", "feed_majority")):
            raise ValueError("Invalid Qill or J")
    for field, value in result["summary"].items():
        expected = sum(row[field] for row in metrics[-settings["tail_rounds"]:]) / settings["tail_rounds"]
        if not math.isfinite(value) or not math.isclose(expected, value, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"Tail summary mismatch: {field}")
    if not {"perceived_majority", "feed_majority", "advocate_perception_bias",
            "public_conformity", "opponent_silence"} <= result["summary"].keys():
        raise ValueError("Required summary fields missing")


def completed_result(prepared, manifest, task, runtime):
    directory = run_directory(prepared, manifest, task)
    if read_json(directory / "identity.json") != task_identity(manifest, task, runtime):
        raise ValueError("Run identity mismatch")
    seal = read_json(directory / "complete.json")
    expected_files = {"result.json", "trajectory.jsonl.gz", "calls.jsonl"}
    if (set(seal["files"]) != expected_files or seal["bundle_id"] != manifest["bundle_id"]
            or seal["task_id"] != task["id"]):
        raise ValueError("Completion seal mismatch")
    for filename, expected in seal["files"].items():
        if file_hash(directory / filename) != expected:
            raise ValueError(f"Completed file changed: {task['id']}/{filename}")
    result = read_json(directory / "result.json")
    baseline = read_json(inside(prepared, manifest["baseline_file"]))
    validate_result(result, task, baseline, task_settings(baseline, manifest["protocol"], task))
    return result


def execute_task(prepared, manifest, task, runtime):
    from scripts.majority_study import run_one
    directory = run_directory(prepared, manifest, task)
    identity = task_identity(manifest, task, runtime)
    if directory.exists():
        if not (directory / "identity.json").exists() or read_json(directory / "identity.json") != identity:
            raise ValueError("Existing run is missing the expected identity; refusing reuse")
        if (directory / "complete.json").exists():
            completed_result(prepared, manifest, task, runtime)
            return
    else:
        directory.mkdir(parents=True)
        write_json(directory / "identity.json", identity)
    baseline = read_json(inside(prepared, manifest["baseline_file"]))
    settings = task_settings(baseline, manifest["protocol"], task)
    if (directory / "result.json").exists():
        # Crash between runner completion and our seal: validate before finalizing.
        result = read_json(directory / "result.json")
        validate_result(result, task, baseline, settings)
    else:
        partial = directory / "trajectory.partial.jsonl.gz"
        if partial.exists():
            attempts = directory / "attempts"
            attempts.mkdir(exist_ok=True)
            index = 1
            while (attempts / f"partial-{index:04d}.jsonl.gz").exists():
                index += 1
            partial.rename(attempts / f"partial-{index:04d}.jsonl.gz")
        network = load_network(inside(prepared, task["network_file"]))
        profiles = tuple(Profile(**row) for row in read_json(inside(prepared, task["profiles_file"])))
        positions = read_json(inside(prepared, task["positions_file"]))
        if (positions["graph"], positions["repeat"], positions["k"]) != (task["graph"], task["repeat"], task["k"]):
            raise ValueError("Position identity mismatch")
        _, bank = load_bank(inside(prepared, manifest["bank_file"]))
        result = run_one(network, profiles, frozenset(positions["positions"]["TH"]), bank,
                         settings, task["group_seed"], "TH", directory, True)
        validate_result(result, task, baseline, settings)
    write_json(directory / "complete.json", {
        "bundle_id": manifest["bundle_id"], "task_id": task["id"],
        "files": {name: file_hash(directory / name)
                  for name in ("result.json", "trajectory.jsonl.gz", "calls.jsonl")}})


def run_batch(prepared, batch, *, allow_api=False):
    if not allow_api:
        raise ValueError("API execution is opt-in: review preparation/budget and supply --allow-api")
    prepared = Path(prepared).resolve()
    manifest = load_prepared(prepared)
    from majority.llm import canonical_model, load_settings
    config, _ = load_settings(PROJECT / ".env")
    if canonical_model(config.model) != "deepseek-flash":
        raise ValueError("Unexpected model: changing models requires a new protocol")
    runtime = {"llm": {**asdict(config), "timeout": 180}, "temperature": 0, "thinking": "disabled",
               "packages": {name: importlib.metadata.version(name) for name in ("numpy", "openai", "python-dotenv")}}
    runtime_path = prepared / "runtime.json"
    if runtime_path.exists():
        if read_json(runtime_path) != runtime:
            raise ValueError("Model, gateway, limits, or packages changed; refusing mixed-batch recovery")
    else:
        write_json(runtime_path, runtime)
    tasks = manifest["tasks"]
    if batch == "pilot":
        selected = [task for task in tasks if task["pilot"]]
    elif batch == "remaining":
        for task in tasks:
            if task["pilot"]:
                completed_result(prepared, manifest, task, runtime)
        selected = [task for task in tasks if not task["pilot"]]
    else:
        raise ValueError("Choose pilot or remaining; the pilot must be reviewed first")

    def execute(task):
        try:
            execute_task(prepared, manifest, task, runtime)
            return {"task": task["id"], "status": "complete"}
        except Exception as error:
            # No raw exception bodies, payloads or credentials in this status summary.
            return {"task": task["id"], "status": "failed", "error_type": type(error).__name__}

    with ThreadPoolExecutor(max_workers=manifest["protocol"]["execution"]["run_workers"]) as pool:
        statuses = list(pool.map(execute, selected))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    write_json(prepared / f"batch-{batch}-{stamp}.json", statuses)
    failed = [row for row in statuses if row["status"] != "complete"]
    print(json.dumps({"selected": len(selected), "failed": failed}, ensure_ascii=False))
    if failed:
        raise RuntimeError("Some tasks failed; no missing result was substituted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="Offline only; freezes inputs and searches positions")
    preparation.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    preparation.add_argument("--output", type=Path, required=True)
    execution = commands.add_parser("run", help="Explicitly authorized API execution only")
    execution.add_argument("--prepared", type=Path, required=True)
    execution.add_argument("--batch", choices=("pilot", "remaining"), required=True)
    execution.add_argument("--allow-api", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.protocol, args.output)
    else:
        run_batch(args.prepared, args.batch, allow_api=args.allow_api)


if __name__ == "__main__":
    main()
