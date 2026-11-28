"""Read-only collection, then explicit new-directory summaries of main experiment 2.

No API calls. Full inference requires all planned conditions; --pilot reports only
technical completeness, descriptive values and usage, without hypothesis tests.
"""

import argparse
import gzip
import itertools
import json
import math
from pathlib import Path
from statistics import mean

from scripts.majority_factorial import (
    PRIMARY_COMPARISONS, completed_result, digest, file_hash, inside, load_prepared,
    read_json, run_directory, write_json,
)


def primary_contrasts(values, k_grid, rho_grid):
    """One group's full condition map -> count, activity and interaction contrasts."""
    low_k, high_k = min(k_grid), max(k_grid)
    low_rho, high_rho = min(rho_grid), max(rho_grid)
    count = mean(values[high_k, rho] - values[low_k, rho] for rho in rho_grid)
    activity = mean(values[k, high_rho] - values[k, low_rho] for k in k_grid)
    interaction = ((values[high_k, high_rho] - values[low_k, high_rho])
                   - (values[high_k, low_rho] - values[low_k, low_rho]))
    return count, activity, interaction


def holm(pvalues):
    adjusted = [0.0] * len(pvalues)
    previous = 0.0
    for rank, index in enumerate(sorted(range(len(pvalues)), key=pvalues.__getitem__)):
        previous = max(previous, min(1.0, pvalues[index] * (len(pvalues) - rank)))
        adjusted[index] = previous
    return adjusted


def analyze_rows(rows, protocol):
    """Complete groups, graph-stratified paired bootstrap, three predeclared tests."""
    import numpy as np

    ks, rhos = protocol["k_grid"], protocol["rho_grid"]
    groups = [(graph, repeat) for graph in protocol["graphs"] for repeat in protocol["repeats"]]
    conditions = [(k, rho) for k in ks for rho in rhos]
    expected = {(graph, repeat, k, rho) for graph, repeat in groups for k, rho in conditions}
    indexed = {}
    for row in rows:
        key = row["graph"], row["repeat"], row["k"], row["rho"]
        if key in indexed:
            raise ValueError(f"Duplicate condition: {key}")
        value = row["q_tail"]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid outcome")
        indexed[key] = value
    if set(indexed) != expected:
        raise ValueError(f"Incomplete grid: missing={len(expected - set(indexed))}, extra={len(set(indexed) - expected)}")
    descriptive = protocol["analysis"].get("mode") == "three_graph_descriptive"
    if descriptive and (protocol["graphs"] != [0, 1, 2] or protocol["repeats"] != [0]):
        raise ValueError("Descriptive V2 analysis requires three graphs with r0 only")
    if not descriptive and (len(groups) > 20 or len(protocol["repeats"]) < 2):
        raise ValueError("Exact sign-flip inference requires <=20 groups and >=2 repeats per graph")
    values = np.array([[indexed[graph, repeat, k, rho] for k, rho in conditions]
                       for graph, repeat in groups], dtype=float)
    contrasts = np.array([primary_contrasts(dict(zip(conditions, row)), ks, rhos) for row in values])
    graph_ids = np.array([group[0] for group in groups])

    def equally_weighted_mean(array):
        return np.mean([array[graph_ids == graph].mean(axis=0) for graph in protocol["graphs"]], axis=0)

    estimates = equally_weighted_mean(contrasts)
    cell_means = equally_weighted_mean(values)
    if descriptive:
        cells = [{"k": k, "rho": rho, "mean_q": float(cell_means[i]),
                  "range": [float(values[:, i].min()), float(values[:, i].max())],
                  "graph_values": {str(g): float(values[j, i]) for j, (g, _) in enumerate(groups)}}
                 for i, (k, rho) in enumerate(conditions)]
        comparisons = [{"name": name, "difference": float(estimates[i]),
                        "graph_differences": {str(g): float(contrasts[j, i]) for j, (g, _) in enumerate(groups)}}
                       for i, name in enumerate(PRIMARY_COMPARISONS)]
        threshold = protocol["analysis"]["coverage_target"]
        return {"paired_groups": 3, "mode": "three_graph_descriptive", "cells": cells,
                "comparisons": comparisons, "coverage_target": threshold,
                "smallest_reached_k_in_tested_grid": {
                    str(rho): next((k for k in ks if cells[conditions.index((k, rho))]["mean_q"] >= threshold), None)
                    for rho in rhos},
                "inference_scope": "Three fixed networks, r0 profiles and inherited TH positions; descriptive only",
                "intervals": "Observed network ranges are not confidence intervals; no significance claims"}
    rng = np.random.default_rng(protocol["analysis"]["seed"])
    draws, cell_draws = [], []
    graph_indexes = [np.flatnonzero(graph_ids == graph) for graph in protocol["graphs"]]
    for _ in range(protocol["analysis"]["bootstrap_repeats"]):
        indexes = [rng.choice(group, size=len(group), replace=True) for group in graph_indexes]
        draws.append(np.mean([contrasts[group].mean(axis=0) for group in indexes], axis=0))
        cell_draws.append(np.mean([values[group].mean(axis=0) for group in indexes], axis=0))
    alpha = (1 - protocol["analysis"]["confidence_level"]) / 2
    intervals = np.quantile(draws, [alpha, 1 - alpha], axis=0)
    cell_intervals = np.quantile(cell_draws, [alpha, 1 - alpha], axis=0)
    signs = np.array(list(itertools.product((-1, 1), repeat=len(groups))))
    weights = np.array([1 / (len(protocol["graphs"]) * sum(graph_ids == graph)) for graph in graph_ids])
    pvalues = [float(np.mean(abs((signs * contrasts[:, i] * weights).sum(axis=1)) >= abs(estimates[i]) - 1e-12))
               for i in range(len(PRIMARY_COMPARISONS))]
    corrected = holm(pvalues)
    comparisons = [{"name": name, "difference": float(estimates[i]),
                    "ci": intervals[:, i].tolist(), "p_value": pvalues[i], "p_holm": corrected[i],
                    "graph_differences": {str(graph): float(contrasts[graph_ids == graph, i].mean())
                                          for graph in protocol["graphs"]}}
                   for i, name in enumerate(PRIMARY_COMPARISONS)]
    cells = [{"k": k, "rho": rho, "mean_q": float(cell_means[i]), "ci": cell_intervals[:, i].tolist()}
             for i, (k, rho) in enumerate(conditions)]
    threshold = protocol["analysis"]["coverage_target"]
    reached = {str(rho): next((k for k in ks if cells[conditions.index((k, rho))]["mean_q"] >= threshold), None)
               for rho in rhos}
    return {"paired_groups": len(groups), "cells": cells, "comparisons": comparisons,
            "coverage_target": threshold, "smallest_reached_k_in_tested_grid": reached,
            "inference_scope": "profile/process groups conditional on fixed networks and TH selection strategy",
            "test_assumption": "group differences are sign-exchangeable under the null",
            "intervals": "pointwise bootstrap intervals, not multiplicity-adjusted simultaneous intervals"}


def summarize_calls(path):
    """Never count replayed audit rows as newly made requests; retain unknown usage."""
    counts = {"actual_calls": 0, "reused_rows": 0, "ok": 0, "api_error": 0,
              "invalid_response": 0, "other_status": 0, "known_input_tokens": 0,
              "known_output_tokens": 0, "calls_without_complete_usage": 0}
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("kind") != "llm_call":
                continue
            call = row["data"]
            if call.get("reused", False):
                counts["reused_rows"] += 1
                continue
            counts["actual_calls"] += 1
            status = call.get("status")
            counts[status if status in ("ok", "api_error", "invalid_response") else "other_status"] += 1
            usage = (call.get("response") or {}).get("usage") or {}
            input_tokens, output_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
            counts["known_input_tokens"] += input_tokens or 0
            counts["known_output_tokens"] += output_tokens or 0
            if input_tokens is None or output_tokens is None:
                counts["calls_without_complete_usage"] += 1
    return counts


def collect_rows(prepared, manifest, pilot=False):
    runtime = read_json(Path(prepared) / "runtime.json")
    rows, incomplete = [], []
    for task in manifest["tasks"]:
        if pilot and not task["pilot"]:
            continue
        directory = run_directory(prepared, manifest, task)
        if not (directory / "complete.json").exists():
            incomplete.append(task["id"])
            continue
        result = completed_result(prepared, manifest, task, runtime)
        positions = read_json(inside(prepared, task["positions_file"]))
        rows.append({"task_id": task["id"], "graph": task["graph"], "repeat": task["repeat"],
                     "k": task["k"], "rho": task["rho"],
                     "q_tail": result["summary"]["perceived_majority"],
                     "summary": result["summary"], "metrics": result["metrics"],
                     "position_diagnostics": positions["audit"]["diagnostics"]["TH"],
                     "result_sha256": file_hash(directory / "result.json"),
                     "usage": summarize_calls(directory / "calls.jsonl")})
    return rows, incomplete


def plot_results(rows, analysis, output, protocol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ks, rhos = protocol["k_grid"], protocol["rho_grid"]
    lookup = {(cell["k"], cell["rho"]): cell["mean_q"] for cell in analysis["cells"]}
    matrix = np.array([[lookup[k, rho] * 100 for k in ks] for rho in rhos])
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    mesh = ax.imshow(matrix, vmin=0, vmax=100, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(ks)), ks)
    ax.set_yticks(range(len(rhos)), rhos)
    ax.set(xlabel="Advocate count k", ylabel="Speech odds multiplier rho",
           title="Observer majority misperception: rounds 8-10 (%)")
    for i, rho in enumerate(rhos):
        for j, k in enumerate(ks):
            ax.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center",
                    color="white" if matrix[i, j] > 55 else "black")
    fig.colorbar(mesh, ax=ax, label="Qill (%)")
    fig.savefig(output / "count_activity_heatmap.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, len(rhos), figsize=(4.5 * len(rhos), 7), squeeze=False,
                             sharex=True, sharey="row", constrained_layout=True)
    for column, rho in enumerate(rhos):
        for k in ks:
            subset = [row for row in rows if row["k"] == k and row["rho"] == rho]
            for axis, field in ((axes[0, column], "perceived_majority"), (axes[1, column], "feed_majority")):
                series = [mean(mean(row["metrics"][i][field] for row in subset if row["graph"] == graph)
                               for graph in protocol["graphs"]) for i in range(10)]
                axis.plot(range(1, 11), np.array(series) * 100, label=f"k={k}", marker="o", markersize=3)
                axis.set_ylim(0, 100)
                axis.grid(alpha=.2)
        axes[0, column].set_title(f"rho={rho}")
        axes[1, column].set_xlabel("Round")
    axes[0, 0].set_ylabel("Qill (%)")
    axes[1, 0].set_ylabel("J (%)")
    axes[0, 0].legend()
    fig.savefig(output / "time_trajectories.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(protocol["graphs"]), figsize=(14, 4), squeeze=False,
                             constrained_layout=True)
    for ax, graph in zip(axes[0], protocol["graphs"]):
        matrix = np.array([[mean(row["q_tail"] for row in rows
                           if row["graph"] == graph and row["k"] == k and row["rho"] == rho)
                           * 100 for k in ks] for rho in rhos])
        mesh = ax.imshow(matrix, vmin=0, vmax=100, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(ks)), ks)
        ax.set_yticks(range(len(rhos)), rhos)
        ax.set(title=f"Network {graph + 1}: rounds 8-10 (%)", xlabel="Advocate count k", ylabel="rho")
        for i in range(len(rhos)):
            for j in range(len(ks)):
                ax.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center",
                        color="white" if matrix[i,j] > 55 else "black")
    fig.colorbar(mesh, ax=list(axes[0]), label="Qill (%)")
    fig.savefig(output / "per_network_heatmaps.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for graph in protocol["graphs"]:
        ax.plot(range(3), [100 * c["graph_differences"][str(graph)] for c in analysis["comparisons"]],
                marker="o", label=f"Network {graph + 1}")
    ax.plot(range(3), [100*c["difference"] for c in analysis["comparisons"]],
            color="black", marker="D", linewidth=2, label="Equal-network mean")
    ax.set_xticks(range(3), ["Count effect", "Activity effect", "Interaction"])
    ax.set(ylabel="Difference (percentage points)", title="Paired endpoint contrasts (descriptive)")
    ax.axhline(0, color="grey", linewidth=1)
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(output / "paired_effects.png", dpi=180)
    plt.close(fig)


def summarize(prepared, output, *, pilot=False):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    manifest = load_prepared(prepared)
    rows, incomplete = collect_rows(prepared, manifest, pilot)
    if incomplete:
        raise ValueError(f"Requested grid is incomplete; no partial inference: {incomplete}")
    if not rows:
        raise ValueError("No completed results")
    analysis = None if pilot else analyze_rows(rows, manifest["protocol"])
    output.mkdir(parents=True, exist_ok=False)
    usage = {key: sum(row["usage"][key] for row in rows) for key in rows[0]["usage"]}
    write_json(output / "run_summaries.json", rows)
    write_json(output / "usage.json", usage)
    write_json(output / "analysis_manifest.json", {
        "bundle_id": manifest["bundle_id"], "pilot_only": pilot,
        "input_results_digest": digest({row["task_id"]: row["result_sha256"] for row in rows}),
        "analysis_source_sha256": file_hash(__file__), "protocol": manifest["protocol"]["analysis"]})
    lines = ["# 主实验二技术试运行报告" if pilot else "# 主实验二结果", "",
             f"完整轨迹：{len(rows)}。私人立场固定，仅50名观察者使用LLM。", "",
             "| 条件 | 图/组 | 末3轮Qill | 末3轮J | 初始IG |",
             "| --- | --- | ---: | ---: | ---: |"]
    if manifest["protocol"].get("settings", {}).get("nonobserver_perception") == "observer_mean":
        lines[4:4] = ["本批采用观察节点均值广播：固定50名观察者本轮更新后的p_hat等权平均，"
                      "含空阅读沿用值，供全部非观察节点下一轮决策使用。主指标仍取固定观察集合。", ""]
    for row in rows:
        lines.append(f"| k={row['k']}, rho={row['rho']} | g{row['graph']}-r{row['repeat']} | "
                     f"{row['q_tail']:.2%} | {row['summary']['feed_majority']:.2%} | "
                     f"{row['position_diagnostics']['illusion']:.2%} |")
    if analysis is not None:
        write_json(output / "analysis.json", analysis)
        if analysis.get("mode") == "three_graph_descriptive":
            lines += ["", "## 预定主要比较（描述性）", "", "| 比较 | 平均差值（百分点） | 网络1 | 网络2 | 网络3 |",
                      "| --- | ---: | ---: | ---: | ---: |"]
            for c in analysis["comparisons"]:
                lines.append(f"| {c['name']} | {100*c['difference']:+.2f} | " + " | ".join(
                    f"{100*c['graph_differences'][str(g)]:+.2f}" for g in manifest["protocol"]["graphs"]) + " |")
            lines += ["", "仅3张固定网络、各1组个体配置；展示配对差异和网络间一致性，不计算退化的图内bootstrap区间，不作显著性结论。",
                      "人数效应：k=240减k=100，跨rho平均；积极性效应：rho=6减rho=1，跨k平均；交互：两端人数差在rho=6与rho=1之间的差。"]
        else:
            lines += ["", "## 预定主要比较", "", "| 比较 | 差值（百分点） | 区间（百分点） | 原始p | Holm p |",
                      "| --- | ---: | --- | ---: | ---: |"]
            for comparison in analysis["comparisons"]:
                lo, hi = comparison["ci"]
                lines.append(f"| {comparison['name']} | {100*comparison['difference']:+.2f} | "
                             f"[{100*lo:+.2f}, {100*hi:+.2f}] | {comparison['p_value']:.4f} | {comparison['p_holm']:.4f} |")
            lines += ["", "区间为逐项bootstrap区间；三项主要检验使用Holm校正。"]
        plot_results(rows, analysis, output, manifest["protocol"])
        lines += ["", "![人数与活跃度](count_activity_heatmap.png)", "",
                  "![感知与信息流时间变化](time_trajectories.png)", "",
                  "![各网络结果](per_network_heatmaps.png)", "", "![配对效应](paired_effects.png)", "",
                  "未达覆盖目标的条件记录为未达标；最小人数仅限本次已测网格，不外推为普适阈值。"]
    else:
        lines += ["", "试运行仅检查技术流程、描述性结果与预算，不进行主假设显著性检验，不据结果高低删选条件。"]
    lines += ["", "## 调用用量", "", "```json", json.dumps(usage, ensure_ascii=False, indent=2), "```", "",
              "已排除缓存复用行；缺失usage没有按实际零消耗解释，已知token不等于供应商最终账单。",
              "结果仅针对本次冻结配置、固定网络与继承的TH位置。新版为本地感知、统一11条阅读及Beta-Binomial多评论；历史结果保持原协议。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    summarize(args.prepared, args.output, pilot=args.pilot)


if __name__ == "__main__":
    main()
