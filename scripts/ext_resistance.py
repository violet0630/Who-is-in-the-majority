"""Extension 1: stance-expression parameters under network rules v2."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.ext_common import (V2_SETTINGS, csv_file, frozen_input_hashes, load_frozen_group,
                                paired_condition_inference, persistence_rows_v2, runner_sha256,
                                tail_rows_v2, verify_frozen_inputs, write_archive_manifest, write_json)
from scripts.majority_study import active_bank, load_bank, run_one

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "experiments/2026-11-21-ext-expression"
CONDITIONS = {
    "E0": {"label": "基准 theta_d=-1, kappa=0.5"},
    "E1": {"label": "高表达阻力 theta_d=-2", "theta_d": -2.0},
    "E2": {"label": "低表达阻力 theta_d=0", "theta_d": 0.0},
    "E3": {"label": "强立场一致 kappa=1", "kappa": 1.0},
}
COMPARISONS = [[condition, "E0"] for condition in ("E1", "E2", "E3")]
METRICS = ("perceived_majority", "feed_majority", "public_conformity", "opponent_silence")
BASE_SETTINGS = {**V2_SETTINGS, "alpha": 0.05, "beta": 10, "llm_workers": 2}


def settings_for(condition):
    settings = {**BASE_SETTINGS}
    theta = list(settings["theta"])
    override = CONDITIONS[condition]
    if "theta_d" in override:
        theta[2] = override["theta_d"]
    settings["theta"] = theta
    if "kappa" in override:
        settings["kappa"] = override["kappa"]
    return settings


def prepare():
    protocol = {
        "experiment": "extension-expression-parameters-v2", "date": "2026-11-21",
        "rounds": 15, "tail_rounds": 5,
        "base": "v2 local perception, read_limit=11, max_comments=3, and frozen TH positions",
        "conditions": {name: settings_for(name) for name in CONDITIONS},
        "labels": {name: value["label"] for name, value in CONDITIONS.items()},
        "comparisons": COMPARISONS, "primary_outcomes": list(METRICS),
        "statistics": "paired groups; graph-stratified bootstrap 10000; exact sign-flip; Holm over three expression comparisons",
        "convergence": "early rounds 1-5 versus late rounds 11-15, final-step delta, and mean absolute last-five-step change",
        "new_runs": 36, "frozen_input_sha256": frozen_input_hashes(), "runner_sha256": runner_sha256(),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Expression protocol differs from frozen protocol")
    if not path.exists():
        write_json(path, protocol)
    (PROJECT / "configs/ext_resistance.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")


def run():
    protocol = json.loads((OUTPUT / "protocol.json").read_text(encoding="utf-8"))
    verify_frozen_inputs(protocol)
    _, bank = load_bank(active_bank())
    jobs = [(condition, graph, repeat) for condition in CONDITIONS
            for graph in range(3) for repeat in range(3)]

    def execute(job):
        condition, graph, repeat = job
        group = load_frozen_group(graph, repeat)
        output = OUTPUT / "runs" / f"{condition}-g{graph}-r{repeat}"
        if (output / "result.json").exists():
            return {"condition": condition, "graph": graph, "repeat": repeat,
                    **json.loads((output / "result.json").read_text(encoding="utf-8"))}
        result = run_one(group["network"], group["profiles"], group["advocates"], bank,
                         settings_for(condition), group["group_seed"], "TH", output, True)
        return {"condition": condition, "graph": graph, "repeat": repeat, **result}

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(execute, jobs))
    write_json(OUTPUT / "results.json", results)


def convergence(metrics):
    output = {}
    for field in METRICS:
        values = np.array([row[field] for row in metrics], dtype=float)
        output[field] = {
            "early_mean": float(values[:5].mean()), "late_mean": float(values[-5:].mean()),
            "late_minus_early": float(values[-5:].mean() - values[:5].mean()),
            "final_step_delta": float(values[-1] - values[-2]),
            "mean_abs_last_five_step": float(np.abs(np.diff(values[-5:])).mean()),
        }
    return output


def analyze():
    results = json.loads((OUTPUT / "results.json").read_text(encoding="utf-8"))
    expected = {(condition, graph, repeat) for condition in CONDITIONS
                for graph in range(3) for repeat in range(3)}
    actual = {(row["condition"], row["graph"], row["repeat"]) for row in results}
    if actual != expected or any(len(row["metrics"]) != 15 for row in results):
        raise ValueError("Expression grid is incomplete")
    rows = [{"condition": row["condition"], "graph": row["graph"], "repeat": row["repeat"],
             "summary": {**row["summary"], "perceived_majority": float(np.mean(
                 [m["perceived_majority"] for m in row["metrics"][-5:]]))},
             "metrics": row["metrics"], "convergence": convergence(row["metrics"])}
            for row in results]
    inference = {field: paired_condition_inference(tail_rows_v2(rows, field), COMPARISONS)
                 for field in METRICS}
    persistence = paired_condition_inference(persistence_rows_v2(rows), COMPARISONS)
    trajectories = {condition: np.mean([[m["perceived_majority"] for m in row["metrics"]]
                                         for row in rows if row["condition"] == condition], axis=0).round(4).tolist()
                    for condition in CONDITIONS}
    convergence_summary = {condition: {
        field: {key: float(np.mean([row["convergence"][field][key] for row in rows
                                    if row["condition"] == condition]))
                for key in ("early_mean", "late_mean", "late_minus_early",
                            "final_step_delta", "mean_abs_last_five_step")}
        for field in METRICS} for condition in CONDITIONS}
    write_json(OUTPUT / "inference.json", {"outcomes": inference, "persistence": persistence,
                                             "trajectories": trajectories,
                                             "convergence": convergence_summary,
                                             "scope": "v2 rules, fixed networks/profiles/TH positions"})
    csv_file(OUTPUT / "summary.csv", [
        {"condition": condition, "label": CONDITIONS[condition]["label"],
         **{field: float(np.mean([row["summary"][field] for row in rows
                                  if row["condition"] == condition])) for field in METRICS},
         "q_early": convergence_summary[condition]["perceived_majority"]["early_mean"],
         "q_late": convergence_summary[condition]["perceived_majority"]["late_mean"],
         "q_late_minus_early": convergence_summary[condition]["perceived_majority"]["late_minus_early"],
         "q_final_step_delta": convergence_summary[condition]["perceived_majority"]["final_step_delta"],
         "q_last5_abs_step": convergence_summary[condition]["perceived_majority"]["mean_abs_last_five_step"]}
        for condition in CONDITIONS])
    figures = OUTPUT / "figures"; figures.mkdir(exist_ok=True)
    colors = ["#285879", "#ba5a31", "#4e8b70", "#8a5ca8"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for index, condition in enumerate(CONDITIONS):
        axes[0].plot(range(1, 16), np.array(trajectories[condition]) * 100,
                     marker="o", markersize=3, label=CONDITIONS[condition]["label"], color=colors[index])
        j_values = np.mean([[m["feed_majority"] for m in row["metrics"]]
                            for row in rows if row["condition"] == condition], axis=0) * 100
        axes[1].plot(range(1, 16), j_values, marker="o", markersize=3,
                     label=CONDITIONS[condition]["label"], color=colors[index])
    axes[0].axvspan(1, 5, color="#dfe6eb", alpha=.4); axes[0].axvspan(11, 15, color="#dfe6eb", alpha=.4)
    axes[0].set_ylabel("Qill (%)"); axes[1].set_ylabel("J (%)"); axes[1].set_xlabel("轮次")
    axes[0].set_title("立场表达参数与 15 轮收敛")
    for axis in axes:
        axis.grid(axis="y", alpha=.2); axis.legend(frameon=False, ncol=2)
    fig.savefig(figures / "expression_convergence.png", dpi=180); plt.close(fig)
    lines = ["# 拓展一 v2：立场表达参数与 15 轮收敛", "", "## 口径", "",
             "不再比较私人立场是否更新。条件只改变 theta_d（感知逆风对评论概率的作用）或 kappa（私人立场对评论赞同方向的作用）。",
             "共同使用 v2 根规则、read_limit=11、Beta(1,9) 多评论、15 轮、三张冻结网络、九个 TH 配对组。", "",
             "## 收敛结果", "", "| 条件 | Qill | J | C | S− | 后期−前期 Qill | 最后一步 Qill |", "|---|---:|---:|---:|---:|---:|---:|"]
    import csv
    with (OUTPUT / "summary.csv").open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            lines.append(f"| {row['label']} | {float(row['perceived_majority']):.2%} | {float(row['feed_majority']):.2%} | {float(row['public_conformity']):.2%} | {float(row['opponent_silence']):.2%} | {float(row['q_late_minus_early']):+.2%} | {float(row['q_final_step_delta']):+.2%} |")
    lines += ["", "## 现实解释", "theta_d 代表感到自己处于逆风时是否减少公开评论。kappa 代表私人立场对评论方向的坚持程度。早期、后期和最后一步变化分开报告，避免把单轮波动称为收敛。", "", "![收敛轨迹](figures/expression_convergence.png)"]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_archive_manifest(OUTPUT)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=("prepare", "run", "analyze")); args = parser.parse_args()
    {"prepare": prepare, "run": run, "analyze": analyze}[args.stage]()


if __name__ == "__main__":
    main()
