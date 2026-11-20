"""Extension 3: network environment — fan heterogeneity (s_in) and community isolation (w)."""

import argparse
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from random import Random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from majority.artifacts import save_artifact
from majority.network import network_statistics
from scripts.ext_common import (FROZEN_SETTINGS, csv_file, frozen_input_hashes, load_frozen_group,
                                mean_summary, persistence_rows, pp, runner_sha256, tail_rows,
                                unpaired_condition_inference, verify_frozen_inputs,
                                write_archive_manifest, write_json)
from scripts.majority_study import (active_bank, draw_profiles, load_bank, make_network,
                                    network_config, run_one, save_json, select_positions)

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "experiments/2026-11-20-ext-network"

NETWORK_CONDITIONS = {"NW1": {"s_in": 0.5}, "NW2": {"s_in": 1.5},
                      "NW3": {"w": 0.05}, "NW4": {"w": 0.5}}
COMPARISONS = [["NW1", "NW0"], ["NW2", "NW0"], ["NW3", "NW0"], ["NW4", "NW0"]]
COND_LABELS = {"NW0": "基准 s_in=1.0, w=0.2", "NW1": "粉丝差异↓ s_in=0.5", "NW2": "粉丝差异↑ s_in=1.5",
               "NW3": "社群隔离↑ w=0.05", "NW4": "社群隔离↓ w=0.5"}
INDEX = {"NW1": 1, "NW2": 2, "NW3": 3, "NW4": 4}
BASE_SETTINGS = {**FROZEN_SETTINGS, "alpha": 0.05, "beta": 10}
ALL_CONDITIONS = {"NW0": {}, **NETWORK_CONDITIONS}
METRIC_LABELS = {"perceived_majority": "多数误判 Qill", "feed_majority": "信息流多数 J",
                 "public_conformity": "公开顺从 C", "opponent_silence": "反对者沉默 S−"}
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False})


def network_seed(condition: str, graph: int) -> int:
    return 41000 + INDEX[condition] * 10 + graph


def profile_seed(condition: str, graph: int, repeat: int) -> int:
    return 42000 + INDEX[condition] * 100 + graph * 10 + repeat


def search_seed(condition: str, graph: int, repeat: int) -> int:
    return 43000 + INDEX[condition] * 100 + graph * 10 + repeat


def group_seed(condition: str, graph: int, repeat: int) -> int:
    return 44000 + INDEX[condition] * 100 + graph * 10 + repeat


def condition_settings(condition: str) -> dict:
    return {**BASE_SETTINGS, **NETWORK_CONDITIONS[condition]}


def prepare():
    protocol = {
        "experiment": "extension-3 network environment", "date": "2026-11-20",
        "base": "main experiment 1 frozen settings (alpha=0.05, beta=10 explicit); TH positions; one factor at a time",
        "conditions": {"NW0": BASE_SETTINGS,
                       **{name: condition_settings(name) for name in NETWORK_CONDITIONS}},
        "labels": COND_LABELS, "primary_outcome": "mean observer perceived_majority over rounds 8-10",
        "secondary_outcomes": list(METRIC_LABELS),
        "comparisons": COMPARISONS,
        "statistics": "unpaired: 9 runs per condition (3 networks x 3 groups); graph-stratified bootstrap 10000; "
                      "shifted-null two-sided p (ties not double-counted); Holm (4 per outcome; persistence Holm "
                      "separately); inference conditional on the 12 generated networks (4 conditions x 3 graphs)",
        "baseline_policy": "NW0 rerun on the archived networks with current runner and gateway; the archived "
                           "2026-11-16 TH results are a descriptive reference only",
        "new_runs": 45,
        "position_search": "TH via select_positions steps=20000 restarts=4 rho=6",
        "seed_formulas": {"network": "41000+cond*10+graph", "profiles": "42000+cond*100+graph*10+repeat",
                          "search": "43000+cond*100+graph*10+repeat", "group": "44000+cond*100+graph*10+repeat"},
        "frozen_input_sha256": frozen_input_hashes(),
        "runner_sha256": runner_sha256(),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == protocol
    else:
        save_json(path, protocol)
    (PROJECT / "configs/ext_network.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")

    for name in NETWORK_CONDITIONS:
        settings = condition_settings(name)
        for graph in range(3):
            network_path = OUTPUT / "inputs" / f"{name}-network-{graph}.json"
            if network_path.exists():
                continue
            network = make_network(1000, settings, network_seed(name, graph))
            save_artifact(network_path, "network", {"seed": network_seed(name, graph),
                "config": network_config(1000, settings), "network": network,
                "statistics": network_statistics(network),
                "sampler": "rowwise rejection conditional on positive outdegree"})
        print(f"{name} networks ready", flush=True)

    def prepare_group(job):
        name, graph, repeat = job
        from majority.artifacts import load_network
        network = load_network(OUTPUT / "inputs" / f"{name}-network-{graph}.json")
        settings = condition_settings(name)
        prepared_path = OUTPUT / "inputs" / f"{name}-g{graph}-r{repeat}_positions.json"
        if prepared_path.exists():
            return None
        profiles = draw_profiles(1000, Random(profile_seed(name, graph, repeat)))
        positions, diagnostics = select_positions(network, profiles, 200,
            search_seed(name, graph, repeat), steps=20000, restarts=4, rho=settings["rho"])
        save_json(prepared_path, {"condition": name, "graph": graph, "repeat": repeat,
            "profiles": [p.__dict__ for p in profiles],
            "positions": {b: sorted(nodes) for b, nodes in positions.items()},
            "diagnostics": diagnostics})
        return f"{name}-g{graph}-r{repeat}"

    jobs = [(name, graph, repeat) for name in NETWORK_CONDITIONS for graph in range(3) for repeat in range(3)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for done in pool.map(prepare_group, jobs):
            if done:
                print(f"prepared {done}", flush=True)
    print("network preparation complete", flush=True)


def run():
    protocol = json.loads((OUTPUT / "protocol.json").read_text(encoding="utf-8"))
    verify_frozen_inputs(protocol)
    _, bank = load_bank(active_bank())
    jobs = []
    for name in ALL_CONDITIONS:
        settings = protocol["conditions"][name]
        for graph in range(3):
            for repeat in range(3):
                jobs.append((name, graph, repeat, settings, bank))

    def execute(job):
        name, graph, repeat, settings, bank = job
        output = OUTPUT / "runs" / f"{name}-g{graph}-r{repeat}"
        if (output / "result.json").exists():
            return {"condition": name, "graph": graph, "repeat": repeat,
                    **json.loads((output / "result.json").read_text(encoding="utf-8"))}
        if name == "NW0":
            group = load_frozen_group(graph, repeat)
            result = run_one(group["network"], group["profiles"], group["advocates"], bank, settings,
                             group["group_seed"], "TH", output, True)
        else:
            from majority.artifacts import load_network
            network = load_network(OUTPUT / "inputs" / f"{name}-network-{graph}.json")
            prepared = json.loads((OUTPUT / "inputs" / f"{name}-g{graph}-r{repeat}_positions.json")
                                  .read_text(encoding="utf-8"))
            from majority.models import Profile
            profiles = tuple(Profile(**row) for row in prepared["profiles"])
            advocates = frozenset(prepared["positions"]["TH"])
            result = run_one(network, profiles, advocates, bank, settings,
                             group_seed(name, graph, repeat), "TH", output, True)
        return {"condition": name, "graph": graph, "repeat": repeat, **result}

    with ThreadPoolExecutor(max_workers=18) as pool:
        results = list(pool.map(execute, jobs))
    write_json(OUTPUT / "results.json", results)
    return results


def _scope_table(rows):
    """范围：按社团标签分组观察者的末3轮 Qill（需要轨迹回放）。"""
    scope = {}
    for row in rows:
        cond, graph = row["condition"], row["graph"]
        if cond == "NW0":
            from scripts.ext_common import MAIN_ARCHIVE
            from majority.artifacts import load_network
            network = load_network(MAIN_ARCHIVE / "03_positions" / f"network-{graph}.json")
        else:
            from majority.artifacts import load_network
            network = load_network(OUTPUT / "inputs" / f"{cond}-network-{graph}.json")
        observers = network.observers
        trajectory = OUTPUT / "runs" / f"{cond}-g{graph}-r{row['repeat']}" / "trajectory.jsonl.gz"
        values = {}
        with gzip.open(trajectory, "rt", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record["kind"] != "round":
                    continue
                data = record["data"]
                if not 8 <= data["round"] <= 10:
                    continue
                for node_row in data["nodes"]:
                    node = node_row["node"]
                    if node not in observers:
                        continue
                    focal = 1 - node_row["after"]["perceived_support"] > 0.5
                    community = network.communities[node]
                    values.setdefault(community, []).append(float(focal))
        scope[f"{cond}-g{graph}-r{row['repeat']}"] = {
            community: float(np.mean(values[community])) if community in values else None
            for community in range(4)}
    return scope


def analyze():
    results = json.loads((OUTPUT / "results.json").read_text(encoding="utf-8"))
    rows = [{"condition": row["condition"], "graph": row["graph"], "repeat": row["repeat"],
             "summary": row["summary"], "metrics": row["metrics"]} for row in results]
    expected = {(name, graph, repeat) for name in ALL_CONDITIONS
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
    output = {"outcomes": {}, "persistence": None, "trajectories": {}, "per_network": {}, "scope": {}}
    for field in METRIC_LABELS:
        output["outcomes"][field] = unpaired_condition_inference(tail_rows(rows, field), COMPARISONS)
    output["persistence"] = unpaired_condition_inference(persistence_rows(rows), COMPARISONS)
    write_json(OUTPUT / "inference.json", output)
    per_round = {}
    for row in rows:
        per_round.setdefault(row["condition"], []).append(
            [m["perceived_majority"] for m in row["metrics"]])
    output["trajectories"] = {cond: np.mean(per_round[cond], axis=0).round(4).tolist()
                              for cond in per_round}
    write_json(OUTPUT / "trajectories.json", output["trajectories"])
    output["per_network"] = {
        cond: {graph: float(np.mean([row["summary"]["perceived_majority"] for row in rows
                                     if row["condition"] == cond and row["graph"] == graph]))
               for graph in range(3)} for cond in ALL_CONDITIONS}
    output["scope"] = _scope_table(rows)
    write_json(OUTPUT / "per_network.json", {"per_network": output["per_network"],
                                             "scope": output["scope"]})
    csv_file(OUTPUT / "summary.csv", [
        {"condition": cond, "label": COND_LABELS[cond],
         **{field: mean_summary(rows, cond, field) for field in METRIC_LABELS},
         **{f"network_{g}": output["per_network"][cond][g] for g in range(3)}}
        for cond in per_round])
    _figures(rows, per_round, output["per_network"])
    _report(output, rows)
    write_archive_manifest(OUTPUT)
    print(json.dumps(output["outcomes"], ensure_ascii=False, indent=2))


def _figures(rows, per_round, per_network):
    figures = OUTPUT / "figures"
    figures.mkdir(exist_ok=True)
    colors = ["#17669b", "#e28743", "#4e9e88", "#9b59b6", "#c0392b"]
    rng = np.random.default_rng(20261120)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for index, cond in enumerate(ALL_CONDITIONS):
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
    ax.set_title("网络环境条件：9组平均轨迹，阴影为95%置信区间")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(figures / "network_trajectories.png", dpi=180)
    plt.close(fig)
    heat = np.array([[per_network[cond][graph] for graph in range(3)] for cond in ALL_CONDITIONS])
    fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
    mesh = ax.imshow(heat * 100, cmap="Blues", vmin=0, vmax=max(50, heat.max() * 100), aspect="auto")
    ax.set_xticks(range(3), [f"网络 {i+1}" for i in range(3)])
    ax.set_yticks(range(len(ALL_CONDITIONS)), [COND_LABELS[c] for c in ALL_CONDITIONS])
    ax.set_title("各条件×网络误判比例（每格3次重复）")
    for i in range(len(ALL_CONDITIONS)):
        for j in range(3):
            ax.text(j, i, f"{heat[i, j] * 100:.1f}", ha="center", va="center",
                    color="white" if heat[i, j] * 100 > mesh.norm.vmax * 0.6 else "#183b56")
    fig.colorbar(mesh, ax=ax, label="多数误判比例（%）", shrink=0.8)
    fig.savefig(figures / "network_heatmap.png", dpi=180)
    plt.close(fig)


def _report(output, rows):
    lines = ["# 拓展三：网络环境实验结果", "", "2026-11-20", "",
             "## 口径", "",
             "每条件9条10轮轨迹（3张网络×3组参数，TH位置）。网络条件之间网络不同，为非配对比较：",
             "各条件独立图内分层 bootstrap 10000 次，差值分布双侧 p（零值平局不重复计数），每个指标4项Holm校正；",
             "持续性（第8–10轮−第3–5轮）另成4项Holm检验。推断以 12 张生成网络为条件（4个新条件×3张网络）。",
             "NW0 基准在主实验一的三张网络上用当前运行器与网关重跑；2026-11-16归档的TH结果仅作描述性参考。",
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
    lines.extend(["", "## 范围（按社团的观察者 Qill，末3轮平均）", "",
                  "| 条件 | 社团1 | 社团2 | 社团3 | 社团4 |", "|---|---:|---:|---:|---:|"])
    for cond in ALL_CONDITIONS:
        per_community = {}
        for c in range(4):
            entries = [output["scope"][f"{cond}-g{g}-r{r}"][c]
                       for g in range(3) for r in range(3)
                       if f"{cond}-g{g}-r{r}" in output["scope"]
                       and output["scope"][f"{cond}-g{g}-r{r}"].get(c) is not None]
            per_community[c] = float(np.mean(entries)) if entries else None
        lines.append(f"| {COND_LABELS[cond]} | "
                     + " | ".join(percent(v) if v is not None else "—" for v in per_community.values()) + " |")
    lines.extend(["", "![轨迹](figures/network_trajectories.png)", "",
                  "![热图](figures/network_heatmap.png)"])
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def percent(value: float) -> str:
    return f"{100 * value:.1f}%"


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
