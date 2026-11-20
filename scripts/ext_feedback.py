"""Extension 2: feedback mechanism — beta (sensitivity) and alpha (memory speed) conditions."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from scripts.ext_common import (FROZEN_SETTINGS, csv_file, load_frozen_group,
                                load_main_baseline, paired_condition_inference, pp)
from scripts.majority_study import active_bank, load_bank, run_one, save_json

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "experiments/2026-11-20-ext-feedback"

CONDITIONS = {"FB0": {}, "FB1": {"beta": 0}, "FB2": {"beta": 50},
              "FB3": {"alpha": 0.1}, "FB4": {"alpha": 0.5}}
COMPARISONS = [["FB1", "FB0"], ["FB2", "FB0"], ["FB3", "FB0"], ["FB4", "FB0"]]
COND_LABELS = {"FB0": "基准 β=10, α=0.05", "FB1": "β=0 无反馈", "FB2": "β=50 高敏感",
               "FB3": "α=0.1 快速记忆", "FB4": "α=0.5 快速记忆"}


BASE_SETTINGS = {**FROZEN_SETTINGS, "alpha": 0.05, "beta": 10}


def derive_settings(base: dict, condition: str) -> dict:
    return {**base, **CONDITIONS[condition]}


def prepare():
    protocol = {
        "experiment": "extension-2 feedback mechanism", "date": "2026-11-20",
        "base": "main experiment 1 frozen settings (alpha=0.05, beta=10 made explicit); TH positions; one factor at a time",
        "conditions": {name: derive_settings(BASE_SETTINGS, name) for name in CONDITIONS},
        "labels": COND_LABELS, "primary_outcome": "mean observer perceived_majority over rounds 8-10",
        "comparisons": COMPARISONS,
        "statistics": "paired groups identical to main experiment; 10000 graph-stratified bootstrap; "
                      "exact sign-flip; Holm",
        "baseline_source": "experiments/2026-11-16/03_positions/g{g}-r{r}-TH/result.json (not rerun)",
        "new_runs": 36,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == protocol
    else:
        save_json(path, protocol)
    (PROJECT / "configs/ext_feedback.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")


def run():
    protocol = json.loads((OUTPUT / "protocol.json").read_text(encoding="utf-8"))
    _, bank = load_bank(active_bank())
    jobs = []
    for name, settings in protocol["conditions"].items():
        if name == "FB0":
            continue
        for graph in range(3):
            for repeat in range(3):
                group = load_frozen_group(graph, repeat)
                jobs.append((name, graph, repeat, group, settings, bank))

    def execute(job):
        name, graph, repeat, group, settings, bank = job
        output = OUTPUT / "runs" / f"{name}-g{graph}-r{repeat}"
        if (output / "result.json").exists():
            return {"condition": name, "graph": graph, "repeat": repeat,
                    **json.loads((output / "result.json").read_text(encoding="utf-8"))}
        result = run_one(group["network"], group["profiles"], group["advocates"], bank, settings,
                         group["group_seed"], "TH", output, True)
        return {"condition": name, "graph": graph, "repeat": repeat, **result}

    with ThreadPoolExecutor(max_workers=18) as pool:
        results = list(pool.map(execute, jobs))
    save_json(OUTPUT / "results.json", results)
    return results


def analyze():
    results = json.loads((OUTPUT / "results.json").read_text(encoding="utf-8"))
    baseline = load_main_baseline("FB0")
    rows = baseline + [{"condition": row["condition"], "graph": row["graph"], "repeat": row["repeat"],
                        "summary": row["summary"], "metrics": row["metrics"]} for row in results]
    inference = paired_condition_inference(rows, COMPARISONS)
    save_json(OUTPUT / "inference.json", inference)
    per_round = {}
    for row in rows:
        per_round.setdefault(row["condition"], []).append(
            [m["perceived_majority"] for m in row["metrics"]])
    save_json(OUTPUT / "trajectories.json",
              {cond: np.mean(per_round[cond], axis=0).round(4).tolist() for cond in per_round})
    csv_file(OUTPUT / "summary.csv", [
        {"condition": cond, "label": COND_LABELS[cond],
         "Qill_mean": float(np.mean([row["summary"]["perceived_majority"] for row in rows
                                     if row["condition"] == cond])),
         "J_mean": float(np.mean([row["summary"]["feed_majority"] for row in rows
                                  if row["condition"] == cond])),
         "C_mean": float(np.mean([row["summary"]["public_conformity"] for row in rows
                                  if row["condition"] == cond]))}
        for cond in per_round])
    lines = ["# 拓展二：反馈机制实验结果", "", "2026-11-20", "",
             "| 比较 | 差异（百分点） | 95% CI | p（Holm） |", "|---|---:|---:|---:|"]
    for comp in inference["comparisons"]:
        lines.append(f"| {COND_LABELS[comp['conditions'][0]]} − {COND_LABELS[comp['conditions'][1]]} "
                     f"| {pp(comp['difference'])} | [{pp(comp['ci95'][0])}, {pp(comp['ci95'][1])}] "
                     f"| {comp['p_value']:.4f}（{comp['p_holm']:.4f}） |")
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(inference, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "analyze"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "run":
        run()
    else:
        analyze()


if __name__ == "__main__":
    main()
