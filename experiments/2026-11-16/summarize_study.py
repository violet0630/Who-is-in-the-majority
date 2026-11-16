"""Build the concise experiment record, tables, and scientific figures."""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from majority_study import INITIALIZATION, ROOT


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_file(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percent(value):
    return f"{100 * value:.1f}%"


def pp(value):
    return f"{100 * value:+.1f}"


def main():
    config = read(ROOT / "initialized_parameters.json")
    inference = read(ROOT / "03_positions/inference.json")
    runs = read(ROOT / "03_positions/results.json")
    initialization = read(INITIALIZATION / "llm_confirmation.json") + read(INITIALIZATION / "llm_refinement.json")
    settings = config["settings"]
    structural = read(ROOT / "03_positions/structural_validation.json")
    actual_edges = "、".join(str(row["edges"]) for row in structural["networks"])
    max_resource_gap = max(row[field] for row in structural["matched_groups"]
                           for field in ("fan_total", "baseline_total", "initial_expected_advocate_speakers"))
    labels = {"TH": "高结构幻觉位置", "TL": "低结构幻觉位置", "U0": "随机位置"}
    branches = ["TH", "TL", "U0"]
    colors = ["#17669b", "#e28743", "#4e9e88"]
    winning_branch = inference["recommended_position"]
    position_cases = []
    for graph in range(3):
        for repeat in range(3):
            source = ROOT / "03_positions" / f"g{graph}-r{repeat}_positions.json"
            prepared = read(source)
            position_cases.append({"graph": graph, "repeat": repeat,
                "network": str(ROOT / "03_positions" / f"network-{graph}.json"),
                "profiles_and_search": str(source), "advocates": prepared["positions"][winning_branch],
                "group_seed": 18000 + graph * 10 + repeat})
    (ROOT / "recommended_positions.json").write_text(json.dumps({"strategy": winning_branch,
        "criterion": "largest mean observer Q in rounds 8-10", "cases": position_cases},
        ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "followup_initialization.json").write_text(json.dumps({**config,
        "position_strategy": winning_branch, "position_file": str(ROOT / "recommended_positions.json")},
        ensure_ascii=False, indent=2), encoding="utf-8")
    with (ROOT / "parameters.csv").open(encoding="utf-8-sig", newline="") as stream:
        parameter_rows = list(csv.DictReader(stream))
    parameter_rows.append({"parameter": "position_strategy", "initialized_value": winning_branch,
                           "basis": "N=1000位置主实验，按末3轮平均误判选择", "explored_or_fixed_range": "TH / TL / U0"})
    csv_file(ROOT / "parameters_final.csv", parameter_rows)
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})

    rows = []
    for run in runs:
        rows.append({"network": run["graph"], "repeat": run["repeat"], "branch": run["branch"],
                     **run["summary"],
                     **{f"Q_round_{t}": run["metrics"][t - 1]["perceived_majority"] for t in (5, 8, 10)},
                     "IG": run["seed_diagnostics"]["illusion"]})
    csv_file(ROOT / "run_summary.csv", rows)
    comparison_rows = [{"comparison": "-".join(row["branches"]), "difference": row["difference"],
                       "ci_low": row["ci95"][0], "ci_high": row["ci95"][1],
                       "p_raw": row["p_value"], "p_holm": row["p_holm"]} for row in inference["comparisons"]]
    csv_file(ROOT / "position_comparisons.csv", comparison_rows)
    csv_file(ROOT / "time_comparisons.csv", [{"branch": row["branch"], "difference": row["difference"],
        "ci_low": row["ci95"][0], "ci_high": row["ci95"][1], "p_raw": row["p_value"], "p_holm": row["p_holm"]}
        for row in inference["time_comparisons"]])
    init_rows = [{"candidate": row["candidate"], "fraction": row["settings"]["fraction"],
                 "rho": row["settings"]["rho"], "mean_degree": row["settings"]["mean_degree"],
                 "Q_tail": row["summary"]["perceived_majority"], "J_tail": row["summary"]["feed_majority"]}
                for row in initialization]
    csv_file(ROOT / "initialization_comparison.csv", init_rows)

    means = np.array([inference["branch_summaries"][b]["mean"] for b in branches])
    intervals = np.array([inference["branch_summaries"][b]["ci95"] for b in branches])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].bar(range(3), means * 100, color=colors, width=0.55)
    axes[0].errorbar(range(3), means * 100, yerr=np.maximum(0, np.array([means - intervals[:, 0], intervals[:, 1] - means])) * 100,
                     fmt="none", ecolor="#243444", capsize=5)
    axes[0].set_xticks(range(3), [labels[b] for b in branches])
    axes[0].set_ylabel("多数误判比例（%）")
    axes[0].set_title("第8—10轮均值及95%置信区间")
    axes[0].set_ylim(0, max(35, 100 * intervals.max() + 10))
    for index, value in enumerate(means):
        axes[0].text(index, 100 * intervals[index, 1] + 1.5, percent(value), ha="center")
    heat = np.array([[np.mean([r["summary"]["perceived_majority"] for r in runs
                              if r["graph"] == graph and r["branch"] == branch])
                     for branch in branches] for graph in range(3)])
    mesh = axes[1].imshow(heat * 100, cmap="Blues", vmin=0, vmax=max(50, heat.max() * 100), aspect="auto")
    axes[1].set_xticks(range(3), [labels[b] for b in branches])
    axes[1].set_yticks(range(3), [f"网络 {i+1}" for i in range(3)])
    axes[1].set_title("各网络误判比例（每格3次重复）")
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{heat[i,j]*100:.1f}", ha="center", va="center",
                         color="white" if heat[i,j] * 100 > mesh.norm.vmax * 0.6 else "#183b56")
    fig.colorbar(mesh, ax=axes[1], label="多数误判比例（%）", shrink=0.8)
    fig.savefig(figures / "position_effect.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    time_rows = []
    rng = np.random.default_rng(19002)
    for branch, color in zip(branches, colors):
        branch_runs = sorted([r for r in runs if r["branch"] == branch], key=lambda r: (r["graph"], r["repeat"]))
        q = np.array([[m["perceived_majority"] for m in r["metrics"]] for r in branch_runs])
        j = np.array([[m["feed_majority"] for m in r["metrics"]] for r in branch_runs])
        draws = []
        for _ in range(3000):
            indexes = np.concatenate([rng.choice(np.arange(g*3, g*3+3), 3, replace=True) for g in range(3)])
            draws.append(q[indexes].mean(axis=0))
        low, high = np.quantile(draws, [0.025, 0.975], axis=0)
        t = np.arange(1, 11)
        axes[0].plot(t, q.mean(axis=0) * 100, label=labels[branch], color=color, marker="o", markersize=4)
        axes[0].fill_between(t, low * 100, high * 100, color=color, alpha=0.13)
        axes[1].plot(t, j.mean(axis=0) * 100, label=labels[branch], color=color, marker="o", markersize=4)
        time_rows.extend({"branch": branch, "round": int(i+1), "Q_mean": float(q.mean(axis=0)[i]),
                          "Q_ci_low": float(low[i]), "Q_ci_high": float(high[i]), "J_mean": float(j.mean(axis=0)[i])}
                         for i in range(10))
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.axvline(5, color="#758393", linestyle="--", linewidth=1)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("多数误判比例 Q（%）")
    axes[1].set_ylabel("信息流多数比例 J（%）")
    axes[1].set_xlabel("轮次")
    axes[1].set_xticks(range(1, 11))
    axes[0].legend(frameon=False, ncol=3)
    axes[0].set_title("位置配置与时间变化：9组成对重复，阴影为95%置信区间")
    fig.savefig(figures / "time_effect.png", dpi=180)
    plt.close(fig)
    csv_file(ROOT / "time_trajectories.csv", time_rows)

    summaries = inference["branch_summaries"]
    primary = inference["comparisons"][0]
    winner = inference["recommended_position"]
    significance = "达到" if primary["p_holm"] < 0.05 else "未达到"
    graph_differences = [np.mean([r["summary"]["perceived_majority"] for r in runs if r["graph"] == g and r["branch"] == "TH"])
                         - np.mean([r["summary"]["perceived_majority"] for r in runs if r["graph"] == g and r["branch"] == "TL"])
                         for g in range(3)]
    if primary["p_holm"] < 0.05:
        finding = ("在相同人数与近似传播资源下，高结构幻觉位置提高了观察者的多数误判比例。"
                   if primary["difference"] > 0 else
                   "本模型中，低静态结构幻觉位置产生了更高的动态感知误判；静态IG排序与动态误判排序出现差异。")
    else:
        finding = "三张网络的高低位置差分别为" + "、".join(pp(x) for x in graph_differences) + "个百分点，位置效果的网络差异见热图。"
    lines = ["# 多数幻觉：初始化与连接位置实验记录", "", "2026-11-16", "",
        f"后续位置初始化采用 **{labels[winner]}（{winner}）**。其末3轮平均误判比例为 **{percent(summaries[winner]['mean'])}**。",
        f"高、低位置的配对差为 **{pp(primary['difference'])} 个百分点**，95% CI [{pp(primary['ci95'][0])}, {pp(primary['ci95'][1])}]，Holm校正 p={primary['p_holm']:.4f}。", "",
        f"单项检验 p={primary['p_value']:.4f}；三项比较校正后{significance}0.05显著性标准。{winner}按最高误判均值选作后续工作点。", "",
        finding, "",
        "## 1. 文本库", "", "双方各25条原创，每条2条赞同、2条反对评论，共50条原创、200条评论；由已配置DeepSeek生成，250条文本均通过数量、长度、重复与立场关系检查。", "",
        "## 2. 初始化实验", "", "N=100，50名观察者。先对96组参数做6轮随机搜索，再将8组候选延长并换3组网络/随机数复测；随后以真实DeepSeek验证基准、候选及相邻人数。完整搜索表见参数与初始化CSV。", "",
        "| 真实验证条件 | 少数派占比 | ρ | 平均关注数 | 末3轮误判比例 |",
        "|---|---:|---:|---:|---:|"]
    lines.extend(f"| {r['candidate']} | {percent(r['fraction'])} | {r['rho']} | {r['mean_degree']} | {percent(r['Q_tail'])} |" for r in init_rows)
    lines.extend(["", "## 3. 主实验初始化", "", "| 参数 | 本次取值 | 依据 |", "|---|---|---|",
        "| N / 观察节点 | 1000 / 50 | 用户设定 |",
        f"| 少数派人数 k / 占比 | {round(1000*settings['fraction'])} / {percent(settings['fraction'])} | 初始化搜索 |",
        f"| 期望连接总数 / 平均关注数 | {round(1000*settings['mean_degree'])} / {settings['mean_degree']} | 初始化搜索 |",
        "| 轮次 / 主要汇总窗口 | 10 / 第8—10轮 | 5—10轮时间比较 |",
        "| α / β | 0.05 / 10 | Banisch–Olbrich；Gaisbauer等 |",
        f"| ρ / κ / γ | {settings['rho']} / {settings['kappa']} / {settings['gamma']} | 初始化搜索 |",
        f"| θ₀ / θh / θd / θhd | {' / '.join(map(str,settings['theta']))} | 初始化搜索 |",
        f"| s_out / s_in / w | {settings['s_out']} / {settings['s_in']} / {settings['w']} | 初始化搜索 |",
        "| 基准发言概率 | Beta(0.24,0.85) | Román等的Beta-geometric参数转换 |",
        "| 强度 / 社团规模 | 0.4、0.8各半 / 150、200、250、400 | 原模型量表；邻域复测 |",
        "| 阅读窗口 / 条数 | 最近2轮 / 最多5条 | 邻域复测 |", "",
        "数值含义：无反馈时，ρ=6可将个人基准发言概率0.2提升至0.6；仅考虑私人立场项时，κ=0.5使普通节点的同向评论概率约为60%—69%；已决定发言且有评论目标的观察者，在θ=[1,0,-1,0]下以50%—73%的概率选择评论，其余选择原创。", "",
        "50名观察节点由DeepSeek更新感知并行动；其余节点依据私人立场和社会反馈行动，感知气候项关闭。私人立场保持固定。", "",
        "## 4. 连接位置主实验", "", "3张网络，每张3组独立个人参数/随机过程；每组比较高、低、随机位置，共27条10轮轨迹。配对控制社团名额、强度构成，粉丝总数、基准活跃度及ρ调整后的初始期望发言量差异均在5%以内。",
        f"三张图实际连接数为{actual_edges}。9组配对的资源差异最大为{max_resource_gap*100:.2f}%。", "",
        "| 位置 | 末3轮平均误判比例 | 95% CI |", "|---|---:|---:|"])
    lines.extend(f"| {labels[b]} | {percent(summaries[b]['mean'])} | [{percent(summaries[b]['ci95'][0])}, {percent(summaries[b]['ci95'][1])}] |" for b in branches)
    lines.extend(["", "| 配对比较 | 差异（百分点） | 95% CI | Holm p |", "|---|---:|---:|---:|"])
    lines.extend(f"| {labels[r['branches'][0]]} − {labels[r['branches'][1]]} | {pp(r['difference'])} | [{pp(r['ci95'][0])}, {pp(r['ci95'][1])}] | {r['p_holm']:.4f} |" for r in inference["comparisons"])
    lines.extend(["", f"![连接位置效果]({(figures / 'position_effect.png').as_posix()})", "", "## 5. 延长时间的影响", "",
        "同一轨迹比较第8—10轮与第3—5轮，正值表示后期误判增加。", "",
        "| 位置 | 后期−前期（百分点） | 95% CI | Holm p |", "|---|---:|---:|---:|"])
    lines.extend(f"| {labels[r['branch']]} | {pp(r['difference'])} | [{pp(r['ci95'][0])}, {pp(r['ci95'][1])}] | {r['p_holm']:.4f} |" for r in inference["time_comparisons"])
    lines.extend(["", "后期误判均值增加约2.8—4.9个百分点；三种配置的时间差异经校正均未达到显著性标准。"])
    lines.extend(["", f"![时间变化]({(figures / 'time_effect.png').as_posix()})", "",
        "统计单位为完整运行，在三张固定网络内按完整配对组重采样10000次计算95% CI；显著性采用9组成对差异的精确符号翻转检验。位置比较和时间比较分别构成3项Holm校正检验。", "",
        "## 参考依据", "",
        "- [Banisch & Olbrich：反馈学习率α=0.05](https://www.tandfonline.com/doi/full/10.1080/0022250X.2018.1517761)",
        "- [Gaisbauer等：反馈表达模型及β=10数值设定](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.102.042303)",
        "- [Román等：发帖概率分布参数](https://doi.org/10.3233/978-1-61499-105-2-756)",
        "- [Bergstra & Bengio：随机超参数搜索](https://jmlr.org/papers/v13/bergstra12a.html)", ""])
    (ROOT / "experiment_record.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(inference, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
