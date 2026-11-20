"""Shared helpers for the three extension experiments (gxy branch)."""

import csv
import hashlib
import json
from itertools import product
from pathlib import Path
from random import Random

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
MAIN_ARCHIVE = PROJECT / "experiments/2026-11-16"
FROZEN_SETTINGS = json.loads((PROJECT / "configs/initialized_20261116.json")
                             .read_text(encoding="utf-8"))["settings"]

FROZEN_INPUT_FILES = (
    [MAIN_ARCHIVE / "03_positions" / f"network-{graph}.json" for graph in range(3)]
    + [MAIN_ARCHIVE / "03_positions" / f"g{graph}-r{repeat}_positions.json"
       for graph in range(3) for repeat in range(3)]
    + [MAIN_ARCHIVE / "01_texts_50" / "bank.json"]
)


def frozen_input_hashes() -> dict:
    """sha256 of every reused frozen input; protocols embed and verify these bytes."""
    return {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in FROZEN_INPUT_FILES}


def runner_sha256() -> str:
    return hashlib.sha256((PROJECT / "scripts/majority_study.py").read_bytes()).hexdigest()


def verify_frozen_inputs(protocol: dict) -> None:
    """Assert the frozen inputs on disk match the protocol's recorded hashes."""
    recorded = protocol.get("frozen_input_sha256")
    if recorded is None:
        raise ValueError("protocol.json lacks frozen_input_sha256; prepare must run first")
    current = frozen_input_hashes()
    if current != recorded:
        raise ValueError("Frozen inputs changed since protocol freeze; refusing to run")
    if protocol.get("runner_sha256") != runner_sha256():
        raise ValueError("Runner scripts/majority_study.py changed since protocol freeze")


def write_json(path: Path, data: object) -> None:
    """Derived outputs (results/inference/reports) may be regenerated on rerun."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def load_frozen_group(graph: int, repeat: int) -> dict:
    """Main-experiment frozen group; identical network/profiles/TH positions/group seed."""
    from majority.artifacts import load_network
    from majority.models import Profile
    network = load_network(MAIN_ARCHIVE / "03_positions" / f"network-{graph}.json")
    path = MAIN_ARCHIVE / "03_positions" / f"g{graph}-r{repeat}_positions.json"
    prepared = json.loads(path.read_text(encoding="utf-8"))
    if prepared.get("graph") != graph or prepared.get("repeat") != repeat:
        raise ValueError(f"Positions file {path} does not match requested group {graph}-{repeat}")
    return {"graph": graph, "repeat": repeat, "network": network,
            "profiles": tuple(Profile(**row) for row in prepared["profiles"]),
            "advocates": frozenset(prepared["positions"]["TH"]),
            "group_seed": 18000 + graph * 10 + repeat,
            "diagnostics": prepared["diagnostics"]}


def load_main_baseline(condition: str) -> list[dict]:
    """Archived main-experiment TH runs, tagged as a descriptive reference condition.

    NOT used as the paired/unpaired statistical baseline: the 2026-11-16 archive was
    produced by a different runner revision and a two-gateway model mix, so comparisons
    against it would conflate treatment with code/model drift. Extension baselines are
    rerun with the current code and gateway instead."""
    rows = []
    for graph in range(3):
        for repeat in range(3):
            path = MAIN_ARCHIVE / "03_positions" / f"g{graph}-r{repeat}-TH" / "result.json"
            row = json.loads(path.read_text(encoding="utf-8"))
            assert row["branch"] == "TH" and row["settings"] == FROZEN_SETTINGS, path
            assert row["size"] == 1000 and row["k"] == 200 and row["tail_rounds"] == 3, path
            rows.append({"condition": condition, "graph": graph, "repeat": repeat,
                         "summary": row["summary"], "metrics": row["metrics"],
                         "settings": row["settings"]})
    return rows


def holm(pvalues: list[float]) -> list[float]:
    adjusted = [0.0] * len(pvalues)
    previous = 0.0
    for rank, index in enumerate(sorted(range(len(pvalues)), key=pvalues.__getitem__)):
        previous = max(previous, min(1.0, pvalues[index] * (len(pvalues) - rank)))
        adjusted[index] = previous
    return adjusted


def _condition_values(rows: list[dict], comparisons: list[tuple[str, str]]) -> tuple[dict, list]:
    conditions = sorted({cond for pair in comparisons for cond in pair})
    grouped = {}
    for row in rows:
        if row["condition"] not in conditions:
            continue
        key = (row["graph"], row["repeat"])
        grouped.setdefault(key, {})[row["condition"]] = row["summary"]["perceived_majority"]
    for key, values in grouped.items():
        if set(values) != set(conditions):
            raise ValueError(f"Group {key} lacks a condition value; pairing is incomplete")
    return grouped, conditions


def _sign_flip_null(differences: list[float]) -> np.ndarray:
    """Null distribution of the mean statistic under sign exchangeability of paired differences."""
    count = len(differences)
    return np.array([np.dot(signs, differences) / count
                     for signs in product((-1, 1), repeat=count)])


def paired_condition_inference(rows: list[dict], comparisons: list[tuple[str, str]],
                               *, repeats: int = 10000, seed: int = 42) -> dict:
    """Paired conditions on identical groups: graph-stratified whole-group bootstrap,
    exact sign-flip p-values, Holm adjustment over the pre-specified comparisons."""
    grouped, conditions = _condition_values(rows, comparisons)
    graphs = {}
    for (graph, _), values in grouped.items():
        graphs.setdefault(graph, []).append(values)
    if not graphs or any(len(groups) < 2 for groups in graphs.values()):
        raise ValueError("Each graph requires at least two complete paired groups")
    rng = Random(seed)

    def draw_means(sample):
        flat = [group for groups in sample for group in groups]
        return {cond: float(np.mean([group[cond] for group in flat])) for cond in conditions}

    observed = draw_means(list(graphs.values()))
    comparisons = [tuple(pair) for pair in comparisons]
    draws = {pair: [] for pair in comparisons}
    for _ in range(repeats):
        sample = [[rng.choice(groups) for _ in groups] for groups in graphs.values()]
        means = draw_means(sample)
        for first, second in comparisons:
            draws[(first, second)].append(means[first] - means[second])
    result = []
    for first, second in comparisons:
        differences = [group[first] - group[second] for group in grouped.values()]
        difference = float(np.mean(differences))
        null = _sign_flip_null(differences)
        pvalue = float(np.mean(np.abs(null) >= abs(difference) - 1e-12))
        values = sorted(draws[(first, second)])
        result.append({"conditions": [first, second], "difference": difference,
                       "ci95": [values[int(0.025 * (repeats - 1))], values[int(0.975 * (repeats - 1))]],
                       "p_value": pvalue, "paired_groups": len(grouped)})
    return _finish(result, conditions, repeats, paired=True)


def unpaired_condition_inference(rows: list[dict], comparisons: list[tuple[str, str]],
                                 *, repeats: int = 10000, seed: int = 42) -> dict:
    """Conditions on different networks: independent graph-stratified bootstrap per condition,
    difference distribution, shifted-null two-sided p, Holm adjustment."""
    conditions = sorted({cond for pair in comparisons for cond in pair})
    per_graph = {cond: {} for cond in conditions}
    for row in rows:
        if row["condition"] in conditions:
            per_graph[row["condition"]].setdefault(row["graph"], []).append(
                row["summary"]["perceived_majority"])
    for cond in conditions:
        graphs = per_graph[cond]
        assert set(graphs) == {0, 1, 2}, f"{cond} must cover the three networks"
        assert all(len(groups) == 3 for groups in graphs.values()), f"{cond} needs 3 groups per graph"
    rng = Random(seed)
    draws = {cond: [] for cond in conditions}
    for _ in range(repeats):
        for cond in conditions:
            sample = [rng.choice(per_graph[cond][graph]) for graph in range(3) for _ in range(3)]
            draws[cond].append(float(np.mean(sample)))
    observed = {cond: float(np.mean([group for groups in per_graph[cond].values() for group in groups]))
                for cond in conditions}
    result = []
    for first, second in comparisons:
        difference = observed[first] - observed[second]
        values = sorted(a - b for a, b in zip(draws[first], draws[second]))
        pvalue = (2 * min(sum(v < 0 for v in values), sum(v > 0 for v in values))
                  + sum(v == 0 for v in values)) / repeats
        result.append({"conditions": [first, second], "difference": difference,
                       "ci95": [values[int(0.025 * (repeats - 1))], values[int(0.975 * (repeats - 1))]],
                       "p_value": pvalue, "paired": False})
    return _finish(result, conditions, repeats, paired=False)


def _finish(comparison_rows: list[dict], conditions: list[str], repeats: int, *, paired: bool) -> dict:
    pvalues = [row["p_value"] for row in comparison_rows]
    for row, adjusted in zip(comparison_rows, holm(pvalues)):
        row["p_holm"] = adjusted
    return {"conditions": conditions, "comparisons": comparison_rows, "paired": paired,
            "statistics": f"{repeats} graph-stratified bootstrap draws; 95% CI; Holm adjustment"}


def tail_rows(rows: list[dict], field: str) -> list[dict]:
    """Rows whose inference value becomes the tail-3 mean of the given per-round metric."""
    return [{**row, "summary": {**row["summary"],
             "perceived_majority": float(np.mean([m[field] for m in row["metrics"][-3:]]))}}
            for row in rows]


def persistence_rows(rows: list[dict]) -> list[dict]:
    """Rows whose inference value is rounds 8-10 minus rounds 3-5 mean Qill (持续性)."""
    return [{**row, "summary": {**row["summary"],
             "perceived_majority": float(np.mean([m["perceived_majority"] for m in row["metrics"][-3:]])
                                         - np.mean([m["perceived_majority"] for m in row["metrics"][2:5]]))}}
            for row in rows]


def mean_summary(rows: list[dict], condition: str, field: str) -> float:
    return float(np.mean([row["summary"][field] for row in rows if row["condition"] == condition]))


def write_archive_manifest(root: Path) -> dict:
    """实验归档清单：全部文件的大小与 sha256（与主实验一 archive_manifest.json 同规）。"""
    manifest = {"files": []}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "archive_manifest.json":
            continue
        manifest["files"].append({"path": relative, "bytes": path.stat().st_size,
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    write_json(root / "archive_manifest.json", manifest)
    return manifest


def verify_archive_manifest(root: Path) -> None:
    manifest = json.loads((root / "archive_manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = root / entry["path"]
        assert path.stat().st_size == entry["bytes"], entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], entry["path"]


def parse_attitude(data: dict, allowed: set[str]):
    """拓展一感知输出解析：support/reason/evidence_ids/new_stance 四个字段严格校验。"""
    if set(data) != {"support", "reason", "evidence_ids", "new_stance"}:
        raise ValueError("Exactly support, reason, evidence_ids, new_stance are required")
    support, reason, ids, stance = data["support"], data["reason"], data["evidence_ids"], data["new_stance"]
    if type(support) not in (int, float) or not 0 <= support <= 1:
        raise ValueError("support must be a number between 0 and 1")
    if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 120:
        raise ValueError("reason must contain 1 to 120 characters")
    if not isinstance(ids, list) or len(ids) > 5 or any(not isinstance(x, str) for x in ids):
        raise ValueError("evidence_ids must contain at most five strings")
    if len(set(ids)) != len(ids) or not set(ids) <= allowed:
        raise ValueError(f"Unknown or duplicate evidence IDs: {sorted(set(ids) - allowed)}")
    if type(stance) is not int or stance not in (1, -1):
        raise ValueError("new_stance must be the integer 1 or -1")
    return float(support), reason, tuple(ids), stance


def percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def pp(value: float) -> str:
    return f"{100 * value:+.1f}"


def csv_file(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
