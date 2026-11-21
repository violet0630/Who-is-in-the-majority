"""Replay targeted extension contrasts from frozen archives without calling a model."""

import argparse
import gzip
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import mean

from scripts.ext_common import holm, verify_archive_manifest


PROJECT = Path(__file__).resolve().parents[1]
ARCHIVED = PROJECT / "experiments"
MARKER = "2026-11-20-ext-"
METRICS = ("perceived_majority", "feed_majority", "public_conformity", "opponent_silence")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def assert_equal_metrics(original, replay, run, number):
    for metric, value in replay.items():
        if abs(original[metric] - value) > 1e-12:
            raise ValueError(f"{run}, round {number}: {metric} replay differs")


def replay_trajectory(root, run):
    result = load(root / "runs" / run / "result.json")
    flip_counts = Counter()
    tail = Counter()
    rounds = []
    final_before = final_after = 0.0
    with gzip.open(root / "runs" / run / "trajectory.jsonl.gz", "rt", encoding="utf-8") as source:
        manifest = json.loads(next(source))
        observers = set(manifest["data"]["network"]["observers"])
        advocates = set(manifest["data"]["advocates"])
        if manifest["kind"] != "manifest" or len(observers) != 50 or len(advocates) != 200 or observers & advocates:
            raise ValueError(f"Unexpected frozen population in {run}")
        seen = {}
        for line in source:
            entry = json.loads(line)
            if entry["kind"] != "round":
                continue
            record = entry["data"]
            number = record["round"]
            rounds.append(number)
            nodes = record["nodes"]
            rows = {row["node"]: row for row in nodes if row["node"] in observers}
            if len(nodes) != 1000 or len({row["node"] for row in nodes}) != 1000 or rows.keys() != observers:
                raise ValueError(f"Population changed in {run}, round {number}")
            published = {message["author"]: message["stance"] for message in record["published"]}
            seen.update({message["id"]: message["stance"] for message in record["published"]})
            fixed_cohort = Counter()
            private_supporters = 0
            for node, row in rows.items():
                before, after = row["before"]["stance"], row["after"]["stance"]
                if before != after:
                    flip_counts["toward_oppose" if after == -1 else "toward_support"] += 1
                reading = row["reading"]["read"]
                j = sum(seen[mid] == -1 for mid in reading) > len(reading) / 2
                q = row["after"]["perceived_support"] < 0.5
                fixed_cohort["J"] += j
                fixed_cohort["Q"] += q
                fixed_cohort["J_only"] += j and not q
                fixed_cohort["Q_only"] += q and not j
                fixed_cohort["published_oppose"] += published.get(node) == -1
                fixed_cohort["silent"] += node not in published
                if before == 1:
                    private_supporters += 1
                    fixed_cohort["contrary"] += published.get(node) == -1
                    fixed_cohort["current_supporter_silent"] += node not in published
            assert_equal_metrics(result["metrics"][number - 1], {
                "perceived_majority": fixed_cohort["Q"] / len(observers),
                "feed_majority": fixed_cohort["J"] / len(observers),
                "public_conformity": fixed_cohort["contrary"] / private_supporters,
                "opponent_silence": fixed_cohort["current_supporter_silent"] / private_supporters,
            }, run, number)
            if any(row["before"]["stance"] != row["after"]["stance"] and row["node"] not in observers for row in nodes):
                raise ValueError(f"Non-observer changed private stance in {run}")
            if number >= 8:
                tail.update(fixed_cohort)
            final_before = mean(row["before"]["stance"] == -1 for row in nodes)
            final_after = mean(row["after"]["stance"] == -1 for row in nodes)
    if rounds != list(range(1, 11)):
        raise ValueError(f"Incomplete or duplicate rounds in {run}")
    return {
        "run": run,
        "tail": {key: count / 150 for key, count in sorted(tail.items())},
        "flips": dict(flip_counts),
        "round_10_before_oppose_share": final_before,
        "round_10_after_oppose_share": final_after,
    }


def network_summary(root):
    per_network = load(root / "per_network.json")["per_network"]
    values = {condition: list(by_graph.values()) for condition, by_graph in per_network.items()}
    baseline = values["NW0"]
    sensitivity = {}
    hypothetical_permutation = {}
    for condition, observed in values.items():
        if condition == "NW0":
            continue
        differences = [mean(a for i, a in enumerate(observed) if i != ai)
                       - mean(b for j, b in enumerate(baseline) if j != bj)
                       for ai in range(3) for bj in range(3)]
        sensitivity[condition] = [min(differences), max(differences)]
        pooled = observed + baseline
        target = abs(mean(observed) - mean(baseline))
        null = [abs(mean(pooled[i] for i in indexes) - mean(pooled[i] for i in range(6) if i not in indexes))
                for indexes in combinations(range(6), 3)]
        hypothetical_permutation[condition] = mean(value >= target - 1e-12 for value in null)
    return {"per_network": values, "leave_one_graph_from_each_condition": sensitivity,
            "hypothetical_exchangeable_graph_label_permutation": hypothetical_permutation}


def summarize_replay(rows):
    grouped = {}
    for row in rows:
        key = row["run"].split("-")[0]
        grouped.setdefault(key, []).append(row)
    return {key: {
        "runs": len(members),
        "tail": {field: mean(row["tail"].get(field, 0) for row in members)
                 for field in sorted({f for row in members for f in row["tail"]})},
        "total_flips_toward_oppose": sum(row["flips"].get("toward_oppose", 0) for row in members),
        "total_flips_toward_support": sum(row["flips"].get("toward_support", 0) for row in members),
        "round_10_before_oppose_share": mean(row["round_10_before_oppose_share"] for row in members),
        "round_10_after_oppose_share": mean(row["round_10_after_oppose_share"] for row in members),
    } for key, members in grouped.items()}


def audit():
    roots = {name: ARCHIVED / (MARKER + name) for name in ("feedback", "network", "resistance")}
    for root in roots.values():
        verify_archive_manifest(root)
    feedback = [replay_trajectory(roots["feedback"], f"{condition}-g{graph}-r{repeat}")
                for condition in ("FB0", "FB2", "FB3") for graph in range(3) for repeat in range(3)]
    resistance = [replay_trajectory(roots["resistance"], f"{condition}-g{graph}-r{repeat}")
                  for condition in ("R0", "R1") for graph in range(3) for repeat in range(3)]
    saved = load(roots["resistance"] / "inference.json")["outcomes"]
    rows = [saved[metric]["comparisons"][0] for metric in METRICS]
    correction = {metric: {**row, "p_holm_protocol": p}
                  for metric, row, p in zip(METRICS, rows, holm([row["p_value"] for row in rows]))}
    return {"feedback": summarize_replay(feedback), "resistance": summarize_replay(resistance),
            "network": network_summary(roots["network"]), "resistance_holm_check": correction,
            "verified_archives": 3, "replayed_trajectories": len(feedback) + len(resistance),
            "scope": "Offline descriptive replay and a protocol-vs-artifact Holm audit; no calls or new trajectories."}


def plot_evidence(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False})
    blue, orange, muted = "#285879", "#ba5a31", "#677681"
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.1))
    fig.patch.set_facecolor("#fbfcfd")
    fig.subplots_adjust(left=.07, right=.985, bottom=.26, top=.72, wspace=.37)
    fig.suptitle("同样从 20% 的少数派开始，三种不同的变化", x=.07, y=.96,
                 ha="left", fontsize=20, fontweight="bold")
    fig.text(.071, .85, "区分真实私人立场、实际看见的多数，以及被认为是多数的一方",
             color=muted, fontsize=12)
    for ax in axes:
        ax.set_facecolor("#fbfcfd")
        ax.spines[["right", "top"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#b9c2ca")
        ax.grid(axis="y", color="#dce2e7", alpha=.6, zorder=0)
        ax.set_axisbelow(True)

    ax = axes[0]
    resistance = result["resistance"]
    tallies = [resistance[key]["tail"] for key in ("R0", "R1")]
    against_private = [100 * v["contrary"] for v in tallies]
    public_oppose = [100 * v["published_oppose"] for v in tallies]
    ax.bar([0, 1], against_private, width=.5, color=blue, label="仍持原立场")
    ax.bar([0, 1], [a - b for a, b in zip(public_oppose, against_private)], width=.5,
           bottom=against_private, color=orange, label="已改变私人立场")
    for index, value in enumerate(public_oppose):
        ax.text(index, value + .13, f"{value:.2f}%", ha="center", fontsize=10)
    ax.set(title="A  公开发出原本反对的观点", ylim=(0, 6.25))
    ax.set_xticks([0, 1], ["固定立场", "允许更新"])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -.18), fontsize=9)

    ax = axes[1]
    feedback = result["feedback"]
    for field, offset, color, label in (("J", -.16, blue, "读到少数派占多数 J"),
                                        ("Q", .16, orange, "误以为少数派占多数 Q")):
        values = [100 * (feedback[key]["tail"][field] - feedback["FB0"]["tail"][field])
                  for key in ("FB2", "FB3")]
        positions = [offset, 1 + offset]
        ax.bar(positions, values, width=.29, color=color, label=label)
        for position, value in zip(positions, values):
            ax.text(position, value + (.15 if value > 0 else -.16), f"{value:+.1f}",
                    va="bottom" if value > 0 else "top", ha="center", fontsize=9)
    ax.axhline(0, color=muted, linewidth=.9)
    ax.set(title="B  曝光与判断可以分离", ylabel="相对基准（百分点）", ylim=(-4.45, 5.45))
    ax.set_xticks([0, 1], ["高反馈敏感度", "较快反馈更新"])
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -.18), fontsize=9)

    ax = axes[2]
    values = result["network"]["per_network"]
    for x, key in enumerate(("NW1", "NW0", "NW2")):
        rates = [100 * value for value in values[key]]
        ax.scatter([x - .12, x, x + .12], rates, color=muted, alpha=.8, s=28, zorder=3)
        ax.plot([x - .22, x + .22], [mean(rates)] * 2, color=blue, linewidth=3, zorder=4)
        ax.text(x + .27, mean(rates), f"{mean(rates):.1f}%", va="center", fontsize=9, color=blue)
    ax.set(title="C  粉丝差异与误判", ylabel="误判 Q", ylim=(37, 66), xlim=(-.39, 2.73))
    ax.set_xticks([0, 1, 2], ["低", "基准", "高"])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.text(0, -.24, "每点一张网络，短横线为条件均值", transform=ax.transAxes, color=muted, fontsize=9)
    fig.text(.072, .035, "第 8–10 轮描述性均值。公开发出相反观点以固定的 50 名原观察者为分母。区间与检验见报告。",
             color=muted, fontsize=9)
    fig.savefig(output / "extension_evidence.png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "diagnostics.json").write_bytes(
        (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    plot_evidence(result, args.output)


if __name__ == "__main__":
    main()
