"""Shared helpers for the three extension experiments (gxy branch)."""

import csv
import json
from itertools import product
from pathlib import Path
from random import Random

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
MAIN_ARCHIVE = PROJECT / "experiments/2026-11-16"
FROZEN_SETTINGS = json.loads((PROJECT / "configs/initialized_20261116.json")
                             .read_text(encoding="utf-8"))["settings"]


def load_frozen_group(graph: int, repeat: int) -> dict:
    """Main-experiment frozen group; identical network/profiles/TH positions/group seed."""
    from majority.artifacts import load_network
    from majority.models import Profile
    network = load_network(MAIN_ARCHIVE / "03_positions" / f"network-{graph}.json")
    path = MAIN_ARCHIVE / "03_positions" / f"g{graph}-r{repeat}_positions.json"
    prepared = json.loads(path.read_text(encoding="utf-8"))
    return {"graph": graph, "repeat": repeat, "network": network,
            "profiles": tuple(Profile(**row) for row in prepared["profiles"]),
            "advocates": frozenset(prepared["positions"]["TH"]),
            "group_seed": 18000 + graph * 10 + repeat,
            "diagnostics": prepared["diagnostics"]}


def load_main_baseline(condition: str) -> list[dict]:
    """Archived main-experiment TH runs, tagged as a baseline condition (not rerun)."""
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
        pvalue = 2 * min(sum(v <= 0 for v in values), sum(v >= 0 for v in values)) / repeats
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
