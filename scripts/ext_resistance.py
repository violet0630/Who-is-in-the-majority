"""Extension 1: individual resistance — fixed vs LLM-updated private stance."""

import argparse
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from majority.llm import prompt_hash
from scripts.ext_common import (FROZEN_SETTINGS, csv_file, frozen_input_hashes, load_frozen_group,
                                mean_summary, paired_condition_inference, persistence_rows, pp,
                                runner_sha256, tail_rows, verify_frozen_inputs, write_archive_manifest, write_json)
from scripts.majority_study import active_bank, load_bank, run_one, save_json

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "experiments/2026-11-20-ext-resistance"

CONDITIONS = {"R0": {"private_update": False, "label": "坚持（固定私人立场）"},
              "R1": {"private_update": True, "label": "更新（LLM 每轮更新立场）"}}
BASE_SETTINGS = {**FROZEN_SETTINGS, "alpha": 0.05, "beta": 10}
METRIC_LABELS = {"perceived_majority": "多数误判 Qill", "feed_majority": "信息流多数 J",
                 "public_conformity": "公开顺从 C", "opponent_silence": "反对者沉默 S−"}
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False})


def prepare():
    protocol = {
        "experiment": "extension-1 individual resistance", "date": "2026-11-20",
        "base": "main experiment 1 frozen settings; TH positions; prompt perception_attitude (v2) in both conditions",
        "conditions": CONDITIONS, "primary_outcome": "mean observer perceived_majority over rounds 8-10",
        "secondary_outcomes": list(METRIC_LABELS),
        "comparisons": [["R1", "R0"]],
        "statistics": "paired groups identical to main experiment; 10000 graph-stratified bootstrap; "
                      "exact sign-flip; Holm over 4 outcomes (Qill, J, C, S-) and persistence separately",
        "mechanics": "new_stance applied to observers only; committed with next-round decisions/messages; "
                     "advocates and non-observers stay fixed; R0 parses but ignores new_stance (same prompt)",
        "guard": "true advocate share <= (200+50)/1000 = 25% < 50% so Qill definition holds",
        "new_runs": 18,
        "prompt_hashes": {"perception": prompt_hash("perception"),
                          "perception_attitude": prompt_hash("perception_attitude")},
        "frozen_input_sha256": frozen_input_hashes(),
        "runner_sha256": runner_sha256(),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == protocol
    else:
        save_json(path, protocol)
    (PROJECT / "configs/ext_resistance.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")


def run():
    protocol = json.loads((OUTPUT / "protocol.json").read_text(encoding="utf-8"))
    verify_frozen_inputs(protocol)
    _, bank = load_bank(active_bank())
    jobs = []
    for name in CONDITIONS:
        for graph in range(3):
            for repeat in range(3):
                jobs.append((name, graph, repeat, bank))

    def execute(job):
        name, graph, repeat, bank = job
        group = load_frozen_group(graph, repeat)
        output = OUTPUT / "runs" / f"{name}-g{graph}-r{repeat}"
        if (output / "result.json").exists():
            return {"condition": name, "graph": graph, "repeat": repeat,
                    **json.loads((output / "result.json").read_text(encoding="utf-8"))}
        result = run_one(group["network"], group["profiles"], group["advocates"], bank, BASE_SETTINGS,
                         group["group_seed"], "TH", output, True,
                         attitude=True, private_update=CONDITIONS[name]["private_update"])
        return {"condition": name, "graph": graph, "repeat": repeat, **result}

    with ThreadPoolExecutor(max_workers=18) as pool:
        results = list(pool.map(execute, jobs))
    write_json(OUTPUT / "results.json", results)
    return results


def extract_conversions(trajectory: Path) -> dict:
    """从单条轨迹逐轮统计私人立场转变人数与真实支持率。"""
    per_round = {}
    with gzip.open(trajectory, "rt", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if item["kind"] != "round":
                continue
            data = item["data"]
            round_number = data["round"]
            flipped_to_oppose = flipped_to_support = 0
            for node_row in data["nodes"]:
                before, after = node_row["before"], node_row["after"]
                if after["stance"] == before["stance"]:
                    continue
                if after["stance"] == -1:
                    flipped_to_oppose += 1
                else:
                    flipped_to_support += 1
            true_support = np.mean([node_row["before"]["stance"] == 1 for node_row in data["nodes"]])
            per_round[round_number] = {"flipped_to_oppose": flipped_to_oppose,
                                       "flipped_to_support": flipped_to_support,
                                       "true_support": float(true_support)}
    return per_round


def conversions():
    """私人转变 D 与真实支持率变化：从轨迹回放逐轮统计观察者立场转变。"""
    rows = json.loads((OUTPUT / "results.json").read_text(encoding="utf-8"))
    records = {}
    for row in rows:
        key = f"{row['condition']}-g{row['graph']}-r{row['repeat']}"
        records[key] = extract_conversions(OUTPUT / "runs" / key / "trajectory.jsonl.gz")
    summary = {}
    for cond in CONDITIONS:
        per_unit = {key: value for key, value in records.items() if key.startswith(f"{cond}-")}
        oppose = [np.mean([value[t]["flipped_to_oppose"] for t in range(1, 11)]) for value in per_unit.values()]
        support = [np.mean([value[t]["flipped_to_support"] for t in range(1, 11)]) for value in per_unit.values()]
        final_oppose_share = [1 - value[10]["true_support"] for value in per_unit.values()]
        summary[cond] = {"mean_flipped_to_oppose": float(np.mean(oppose)),
                         "mean_flipped_to_support": float(np.mean(support)),
                         "final_true_oppose_share": float(np.mean(final_oppose_share))}
    write_json(OUTPUT / "conversions.json", {"per_run": records, "summary": summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def analyze():
    results = json.loads((OUTPUT / "results.json").read_text(encoding="utf-8"))
    rows = [{"condition": row["condition"], "graph": row["graph"], "repeat": row["repeat"],
             "summary": row["summary"], "metrics": row["metrics"]} for row in results]
    expected = {(name, graph, repeat) for name in CONDITIONS for graph in range(3) for repeat in range(3)}
    completed = {(row["condition"], row["graph"], row["repeat"]) for row in rows}
    missing = sorted(expected - completed)
    if missing:
        raise ValueError(f"Incomplete run: {len(missing)} trajectories missing, first: {missing[:5]}; "
                         "no imputation is allowed — rerun the missing trajectories first")
    for row in rows:
        if len(row["metrics"]) != 10:
            raise ValueError(f"{row['condition']}-g{row['graph']}-r{row['repeat']} has "
                             f"{len(row['metrics'])} rounds, expected 10")
        if not all(0.2 - 1e-9 <= metric["true_advocate_support"] <= 0.25 + 1e-9
                   for metric in row["metrics"]):
            raise ValueError(f"{row['condition']}-g{row['graph']}-r{row['repeat']}: true advocate share "
                             "outside [0.2, 0.25]; Qill definition check failed")
    output = {"outcomes": {}, "persistence": None, "trajectories": {}}
    for field in METRIC_LABELS:
        output["outcomes"][field] = paired_condition_inference(tail_rows(rows, field), [["R1", "R0"]])
    output["persistence"] = paired_condition_inference(persistence_rows(rows), [["R1", "R0"]])
    write_json(OUTPUT / "inference.json", output)
    per_round = {}
    for row in rows:
        per_round.setdefault(row["condition"], []).append(
            [m["perceived_majority"] for m in row["metrics"]])
    output["trajectories"] = {cond: np.mean(per_round[cond], axis=0).round(4).tolist()
                              for cond in per_round}
    write_json(OUTPUT / "trajectories.json", output["trajectories"])
    csv_file(OUTPUT / "summary.csv", [
        {"condition": cond, "label": CONDITIONS[cond]["label"],
         **{field: mean_summary(rows, cond, field) for field in METRIC_LABELS}}
        for cond in CONDITIONS])
    _figures(per_round)
    _report(output, rows)
    write_archive_manifest(OUTPUT)
    print(json.dumps(output["outcomes"], ensure_ascii=False, indent=2))


def _figures(per_round):
    figures = OUTPUT / "figures"
    figures.mkdir(exist_ok=True)
    colors = ["#17669b", "#e28743"]
    rng = np.random.default_rng(20261120)
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    for index, cond in enumerate(CONDITIONS):
        values = np.array(per_round[cond])
        draws = [values[rng.choice(len(values), len(values), replace=True)].mean(axis=0)
                 for _ in range(3000)]
        low, high = np.quantile(np.array(draws), [0.025, 0.975], axis=0)
        ax.plot(range(1, 11), values.mean(axis=0) * 100, label=CONDITIONS[cond]["label"],
                color=colors[index], marker="o", markersize=4)
        ax.fill_between(range(1, 11), low * 100, high * 100, color=colors[index], alpha=0.12)
    ax.axvline(5, color="#758393", linestyle="--", linewidth=1)
    ax.set_xlabel("轮次")
    ax.set_ylabel("多数误判比例 Q（%）")
    ax.set_title("个体抵抗力条件：9组平均轨迹，阴影为95%置信区间")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(figures / "resistance_trajectories.png", dpi=180)
    plt.close(fig)


def _report(output, rows):
    conversions_summary = None
    conversions_path = OUTPUT / "conversions.json"
    if conversions_path.exists():
        conversions_summary = json.loads(conversions_path.read_text(encoding="utf-8"))["summary"]
    lines = ["# 拓展一：个体抵抗力实验结果", "", "2026-11-20", "",
             "## 口径", "",
             "两条件共用提示词 perception_attitude（v2，输出四字段）；R0 解析但不应用 new_stance，R1 应用。",
             "只有观察者的私人立场允许更新（倡议者与其余950名普通节点立场始终固定），反对派占比上限 (200+50)/1000=25%，",
             "Qill（反对者被误判为多数）的定义全程有效。新立场与下一轮消息一起提交，公开偏离按同一轮私人立场统计。",
             "配对单位与主实验一完全一致；每个指标构成1项配对比较（符号翻转p=Holm p）。",
             "无缺失轨迹；如有缺失会停止分析而不插补。", "",
             "## 主要结果（R1 − R0）", "",
             "| 指标 | 差异（百分点） | 95% CI | p（Holm） |", "|---|---:|---:|---:|"]
    for field, label in METRIC_LABELS.items():
        comp = output["outcomes"][field]["comparisons"][0]
        lines.append(f"| {label} | {pp(comp['difference'])} | [{pp(comp['ci95'][0])}, {pp(comp['ci95'][1])}] "
                     f"| {comp['p_value']:.4f}（{comp['p_holm']:.4f}） |")
    comp = output["persistence"]["comparisons"][0]
    lines.extend(["", "## 持续性（第8–10轮 − 第3–5轮 Qill，R1 − R0）", "",
                  f"差异 {pp(comp['difference'])} 个百分点，95% CI [{pp(comp['ci95'][0])}, {pp(comp['ci95'][1])}]，"
                  f"p={comp['p_value']:.4f}（Holm {comp['p_holm']:.4f}）。"])
    if conversions_summary:
        lines.extend(["", "## 私人转变 D 与真实支持率", "",
                      "| 条件 | 每轮平均转向反对 | 每轮平均转向支持 | 末轮反对派真实占比 |",
                      "|---|---:|---:|---:|"])
        for cond in CONDITIONS:
            summary = conversions_summary[cond]
            lines.append(f"| {CONDITIONS[cond]['label']} | {summary['mean_flipped_to_oppose']:.2f} "
                         f"| {summary['mean_flipped_to_support']:.2f} "
                         f"| {summary['final_true_oppose_share'] * 100:.1f}% |")
    lines.extend(["", "![轨迹](figures/resistance_trajectories.png)"])
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "analyze", "conversions"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "run":
        run()
    elif args.stage == "conversions":
        conversions()
    else:
        analyze()


if __name__ == "__main__":
    main()
