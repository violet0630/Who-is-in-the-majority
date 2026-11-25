"""Read-only checkpoint analysis: show completed conditions, never impute missing runs."""

import argparse
from pathlib import Path

from scripts.main2_v2 import audit_trajectory
from scripts.majority_factorial import inside, load_prepared, read_json, run_directory, write_json
from scripts.summarize_factorial import collect_rows, summarize_calls


def checkpoint(prepared, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    manifest = load_prepared(prepared)
    protocol = manifest["protocol"]
    rows, missing = collect_rows(prepared, manifest)
    output.mkdir(parents=True, exist_ok=False)
    audited = {}
    complete = {r["task_id"] for r in rows}
    usage = None
    for task in manifest["tasks"]:
        directory = run_directory(prepared, manifest, task)
        if (directory / "calls.jsonl").exists():
            counters = summarize_calls(directory / "calls.jsonl")
            usage = counters if usage is None else {k: usage[k] + counters[k] for k in counters}
        if task["id"] not in complete:
            continue
        net = read_json(inside(prepared, task["network_file"]))["data"]["network"]
        advocates = set(read_json(inside(prepared, task["positions_file"]))["positions"]["TH"])
        audited[task["id"]] = audit_trajectory(directory / "trajectory.jsonl.gz",
            {**protocol["settings"], "rounds": 10, "rho": task["rho"], "alpha": .05},
            set(net["observers"]), advocates)
    write_json(output / "checkpoint.json", {"bundle_id": manifest["bundle_id"],
        "completed": len(rows), "planned": len(manifest["tasks"]), "missing": missing,
        "completed_trajectory_audits": audited, "usage_all_attempted_tasks": usage})
    write_json(output / "completed_run_summaries.json", rows)
    ks, rhos = protocol["k_grid"], protocol["rho_grid"]
    graphs = protocol["graphs"]
    lookup = {(r["graph"], r["k"], r["rho"]): r["q_tail"] * 100 for r in rows}
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#dedede")
    fig, axes = plt.subplots(1, len(graphs), figsize=(14, 4.8), constrained_layout=True)
    for ax, graph in zip(axes, graphs):
        values = np.array([[lookup.get((graph, k, rho), np.nan) for k in ks] for rho in rhos])
        mesh = ax.imshow(values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(ks)), ks)
        ax.set_yticks(range(len(rhos)), rhos)
        ax.set(title=f"Network {graph + 1}", xlabel="Advocate count k", ylabel="Activity rho")
        for i in range(len(rhos)):
            for j in range(len(ks)):
                value = values[i, j]
                ax.text(j, i, "Missing" if np.isnan(value) else f"{value:.1f}", ha="center", va="center",
                        color="white" if value > 55 else "black", fontsize=10)
    fig.suptitle(f"INCOMPLETE: {len(rows)}/36 runs; rounds 8-10 Qill (%)\nGrey means missing, not zero")
    fig.colorbar(mesh, ax=list(axes), label="Observer majority misperception (%)")
    fig.savefig(output / "available_network_heatmaps.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(len(graphs), len(rhos), figsize=(14, 10), sharex=True, sharey=True,
                             constrained_layout=True)
    for i, graph in enumerate(graphs):
        for j, rho in enumerate(rhos):
            ax = axes[i, j]
            subset = sorted((r for r in rows if r["graph"] == graph and r["rho"] == rho), key=lambda r:r["k"])
            for row in subset:
                ax.plot(range(1, 11), [100*m["perceived_majority"] for m in row["metrics"]],
                        marker="o", markersize=3, label=f"k={row['k']}", color=f"C{ks.index(row['k'])}")
            ax.set(title=f"Network {graph+1}, rho={rho}", ylim=(0, 100))
            ax.grid(alpha=.2)
            if subset:
                ax.legend(fontsize=8)
            else:
                ax.text(.5, .5, "No complete runs", transform=ax.transAxes, ha="center")
            if i == len(graphs)-1:
                ax.set_xlabel("Round")
            if j == 0:
                ax.set_ylabel("Qill (%)")
    fig.suptitle("INCOMPLETE checkpoint: completed trajectories only; absent lines are missing runs")
    fig.savefig(output / "available_time_trajectories.png", dpi=180)
    plt.close(fig)
    lines = ["# 主实验二 v2 阶段结果（未完成）", "",
             f"计划36次，完整完成{len(rows)}次，未完成{len(missing)}次。所有完整运行均通过逐条轨迹核查。",
             "", "接口在执行中返回HTTP 402 Insufficient Balance，后续任务未完成；这不是最终36次实验报告。",
             "缺失结果不填零，不计算不完整网格的总体效应，不作显著性结论。灰色Missing表示缺失，数值0.0表示真实已完成结果为零。",
             "", "![各网络已完成条件](available_network_heatmaps.png)", "",
             "![已完成条件的时间曲线](available_time_trajectories.png)", "",
             "| 网络 | k | rho | 第8—10轮平均误判率 |", "| --- | ---: | ---: | ---: |"]
    for row in sorted(rows, key=lambda r:(r["graph"],r["k"],r["rho"])):
        lines.append(f"| {row['graph']+1} | {row['k']} | {row['rho']} | {row['q_tail']:.2%} |")
    lines += ["", "## 恢复方式", "", "同一账户余额恢复后，使用原冻结配置执行以下命令；完成结果跳过，未完成运行利用原请求缓存恢复：", "",
              "```powershell", f"python -m scripts.main2_v2 --prepared {prepared.as_posix()} --allow-api", "```", "",
              "全部完成后程序自动生成完整报告、总体热图、时间曲线、分网络热图及配对效应图。",
              "调用量含未完成任务的尝试，详见checkpoint.json；已知token不是供应商最终账单。"]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Checkpoint: {len(rows)}/36 complete; audited {len(audited)}; wrote two figures", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint(args.prepared, args.output)
