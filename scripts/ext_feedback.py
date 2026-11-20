"""Extension 2: feedback mechanism — beta (sensitivity) and alpha (memory speed) conditions."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.ext_common import (FROZEN_SETTINGS, csv_file, frozen_input_hashes, load_frozen_group,
                                load_main_baseline, mean_summary, paired_condition_inference,
                                persistence_rows, pp, runner_sha256, tail_rows,
                                verify_frozen_inputs, write_archive_manifest, write_json)
from scripts.majority_study import active_bank, load_bank, run_one, save_json

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "experiments/2026-11-20-ext-feedback"

CONDITIONS = {"FB0": {}, "FB1": {"beta": 0}, "FB2": {"beta": 50},
              "FB3": {"alpha": 0.1}, "FB4": {"alpha": 0.5}}
COMPARISONS = [["FB1", "FB0"], ["FB2", "FB0"], ["FB3", "FB0"], ["FB4", "FB0"]]
COND_LABELS = {"FB0": "基准 β=10, α=0.05", "FB1": "β=0 无反馈", "FB2": "β=50 高敏感",
               "FB3": "α=0.1 快速记忆", "FB4": "α=0.5 快速记忆"}
BASE_SETTINGS = {**FROZEN_SETTINGS, "alpha": 0.05, "beta": 10}
METRIC_LABELS = {"perceived_majority": "多数误判 Qill", "feed_majority": "信息流多数 J",
                 "public_conformity": "公开顺从 C", "opponent_silence": "反对者沉默 S−"}
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False})


def derive_settings(base: dict, condition: str) -> dict:
    return {**base, **CONDITIONS[condition]}


def prepare():
    protocol = {
        "experiment": "extension-2 feedback mechanism", "date": "2026-11-20",
        "base": "main experiment 1 frozen settings (alpha=0.05, beta=10 explicit); TH positions; one factor at a time",
        "conditions": {name: derive_settings(BASE_SETTINGS, name) for name in CONDITIONS},
        "labels": COND_LABELS, "primary_outcome": "mean observer perceived_majority over rounds 8-10",
        "secondary_outcomes": list(METRIC_LABELS),
        "comparisons": COMPARISONS,
        "statistics": "paired groups identical to main experiment; 10000 graph-stratified bootstrap; "
                      "exact sign-flip; Holm (4 per outcome; persistence Holm separately)",
        "baseline_policy": "FB0 rerun with current runner and gateway; the archived 2026-11-16 TH results "
                           "are a descriptive reference only (two-gateway model mix, older runner)",
        "new_runs": 45,
        "frozen_input_sha256": frozen_input_hashes(),
        "runner_sha256": runner_sha256(),
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
    verify_frozen_inputs(protocol)
    _, bank = load_bank(active_bank())
    jobs = []
    for name, settings in protocol["conditions"].items():
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
    write_json(OUTPUT / "results.json", results)
    return results


def analyze():
    results = json.loads((OUTPUT / "results.json").read_text(encoding="utf-8"))
    protocol = json.loads((OUTPUT / "protocol.json").read_text(encoding="utf-8"))
    rows = [{"condition": row["condition"], "graph": row["graph"], "repeat": row["repeat"],
             "summary": row["summary"], "metrics": row["metrics"]} for row in results]
    expected = {(name, graph, repeat) for name in CONDITIONS
                for graph in range(3) for repeat in range(3)}
    completed = {(row["condition"], row["graph"], row["repeat"]) for row in rows}
    missing = sorted(expected - completed)
    if missing:
        raise ValueError(f"Incomplete run: {len(missing)} trajectories missing, first: {missing[:5]}; "
                         "no imputation is allowed — rerun the missing trajectories first")
    for row in rows:
        if len(row["metrics"]) != 10:
            raise ValueError(f"{row['condition']}-g{row['graph']}-r{row['repeat']} has "
                             f"{len(row['metrics'])} rounds, expected 10")
    output = {"outcomes": {}, "persistence": None, "trajectories": {}}
    for field, label in METRIC_LABELS.items():
        output["outcomes"][field] = paired_condition_inference(tail_rows(rows, field), COMPARISONS)
    output["persistence"] = paired_condition_inference(persistence_rows(rows), COMPARISONS)
    write_json(OUTPUT / "inference.json", output)
    per_round = {}
    for row in rows:
        per_round.setdefault(row["condition"], []).append(
            [m["perceived_majority"] for m in row["metrics"]])
    output["trajectories"] = {cond: np.mean(per_round[cond], axis=0).round(4).tolist()
                              for cond in per_round}
    write_json(OUTPUT / "trajectories.json", output["trajectories"])
    archive = load_main_baseline("FB0-archived")
    csv_file(OUTPUT / "summary.csv", [
        {"condition": cond, "label": COND_LABELS[cond],
         **{field: mean_summary(rows, cond, field) for field in METRIC_LABELS},
         **{f"{field}_archived": mean_summary(archive, "FB0-archived", field)
            for field in METRIC_LABELS}}
        for cond in per_round])
    _figures(rows, per_round)
    _report(output, rows, protocol)
    write_archive_manifest(OUTPUT)
    print(json.dumps(output["outcomes"], ensure_ascii=False, indent=2))


def _figures(rows, per_round):
    figures = OUTPUT / "figures"
    figures.mkdir(exist_ok=True)
    colors = ["#17669b", "#e28743", "#4e9e88", "#9b59b6", "#c0392b"]
    rng = np.random.default_rng(20261120)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for index, cond in enumerate(CONDITIONS):
        values = np.array(per_round[cond])
        draws = [values[rng.choice(len(values), len(values), replace=True)].mean(axis=0)
                 for _ in range(3000)]
        low, high = np.quantile(np.array(draws), [0.025, 0.975], axis=0)
        ax.plot(range(1, 11), values.mean(axis=0) * 100, label=COND_LABELS[cond],
                color=colors[index], marker="o", markersize=4)
        ax.fill_between(range(1, 11), low * 100, high * 100, color=colors[index], alpha=0.12)
    ax.axvline(5, color="#758393", linestyle="--", linewidth=1)
    ax.set_xlabel("轮次")
    ax.set_ylabel("多数误判比例 Q（%）")
    ax.set_title("反馈机制条件：9组平均轨迹，阴影为95%置信区间")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(figures / "feedback_trajectories.png", dpi=180)
    plt.close(fig)


def _report(output, rows, protocol):
    lines = ["# 拓展二：反馈机制实验结果", "", "2026-11-20", "",
             "## 口径", "",
             "配对单位与主实验一完全一致（同网络/个人参数/TH位置/组种子）；每条件9条10轮轨迹。",
             "主要比较预先指定：FB1–FB4 各自对比 FB0，每个指标构成4项Holm校正检验；持续性（第8–10轮−第3–5轮）另成4项Holm检验。",
             "FB0 基准用当前运行器与网关重跑（45条新轨迹）；2026-11-16归档的TH结果仅作描述性参考（当时为双网关双模型标识混合，见 summary.csv 的 _archived 列）。",
             "无缺失轨迹；如有缺失会停止分析而不插补。", "",
             "## 主要结果", "",
             "| 指标 | 比较 | 差异（百分点） | 95% CI | p（Holm） |", "|---|---|---:|---:|---:|"]
    for field, label in METRIC_LABELS.items():
        for comp in output["outcomes"][field]["comparisons"]:
            lines.append(f"| {label} | {COND_LABELS[comp['conditions'][0]]} − {COND_LABELS[comp['conditions'][1]]} "
                         f"| {pp(comp['difference'])} | [{pp(comp['ci95'][0])}, {pp(comp['ci95'][1])}] "
                         f"| {comp['p_value']:.4f}（{comp['p_holm']:.4f}） |")
    lines.extend(["", "## 持续性（后期−前期 Qill）", "", "| 比较 | 差异（百分点） | 95% CI | p（Holm） |",
                  "|---|---:|---:|---:|"])
    for comp in output["persistence"]["comparisons"]:
        lines.append(f"| {COND_LABELS[comp['conditions'][0]]} − {COND_LABELS[comp['conditions'][1]]} "
                     f"| {pp(comp['difference'])} | [{pp(comp['ci95'][0])}, {pp(comp['ci95'][1])}] "
                     f"| {comp['p_value']:.4f}（{comp['p_holm']:.4f}） |")
    lines.extend(["", "![轨迹](figures/feedback_trajectories.png)"])
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


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
