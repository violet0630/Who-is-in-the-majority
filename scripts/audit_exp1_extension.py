"""Audit the frozen supplemental batch from trajectories and API logs."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from statistics import mean

import numpy as np

from scripts.run_exp1_extension import read, require, sha, validate


FIELDS = ("perceived_majority", "feed_majority", "public_conformity", "opponent_silence")


def audit_trajectory(path, expected_setup, observers, advocates, metrics):
    messages = {}
    seen_rounds = []
    expected_calls = set()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        setup = json.loads(next(stream))
        require(setup["kind"] == "manifest", f"Missing trajectory manifest: {path}")
        for key, value in expected_setup.items():
            require(setup["data"][key] == value, f"Trajectory manifest mismatch ({key}): {path}")
        for line in stream:
            record = json.loads(line)
            if record["kind"] != "round":
                continue
            row = record["data"]
            number = row["round"]
            require(number == len(seen_rounds) + 1 and number <= len(metrics), f"Unexpected round: {path}")
            seen_rounds.append(number)
            selected = {n["node"]: n for n in row["nodes"] if n["node"] in observers}
            require(set(selected) == observers and len(row["nodes"]) == 1000
                    and len({n["node"] for n in row["nodes"]}) == 1000, f"Invalid node coverage: {path}")
            require(all(n["before"]["stance"] == n["after"]["stance"]
                        == (-1 if n["node"] in advocates else 1) for n in row["nodes"]),
                    f"Private stance changed: {path}")
            messages.update({m["id"]: m["stance"] for m in row["published"]})
            published = {m["author"]: m["stance"] for m in row["published"]}
            votes = Counter()
            for node, n in selected.items():
                reading = n["reading"]["read"]
                if reading:
                    expected_calls.add((node, number))
                votes["perceived_majority"] += n["after"]["perceived_support"] < 0.5
                votes["feed_majority"] += sum(messages[m] == -1 for m in reading) > len(reading) / 2
                votes["public_conformity"] += published.get(node) == -1
                votes["opponent_silence"] += node not in published
            recorded = metrics[number - 1]
            require(recorded["round"] == number and all(
                abs(recorded[field] - votes[field] / len(observers)) < 1e-10 for field in FIELDS
            ), f"Metric mismatch in round {number}: {path}")
    require(seen_rounds == list(range(1, 11)), f"Incomplete trajectory: {path}")
    return expected_calls


def audit_calls(path, expected, model, run_id):
    successes = Counter()
    usage = Counter()
    models = Counter()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            require(record["kind"] == "llm_call", f"Unexpected audit entry: {path}")
            call = record["data"]
            context, status = call["context"], call["status"]
            event = (context["node"], context["round"])
            require(event in expected and context["run_id"] == run_id and call["model"] == model,
                    f"Call is outside the run's observer events or model: {path}")
            require(status in {"ok", "invalid_response", "api_error"}, f"Unexpected call status: {path}")
            successes[event] += status == "ok"
            usage[status] += 1
            if call.get("reused"):
                usage["reused"] += 1
                continue
            usage["actual_calls"] += 1
            response = call.get("response") or {}
            if response.get("model"):
                models[response["model"]] += 1
            tokens = response.get("usage") or {}
            for key in ("prompt_tokens", "completion_tokens"):
                if tokens.get(key) is None:
                    usage[f"missing_{key}"] += 1
                else:
                    usage[key] += tokens[key]
            cached = (tokens.get("prompt_tokens_details") or {}).get("cached_tokens")
            if cached is None:
                cached = tokens.get("prompt_cache_hit_tokens")
            if cached is None:
                usage["missing_cached_prompt_tokens"] += 1
            else:
                usage["cached_prompt_tokens"] += cached
    require(set(successes) == expected and all(count >= 1 for count in successes.values()),
            f"Unresolved observer calls: {path}")
    return dict(usage), dict(models)


def paired_q(rows, protocol):
    comparisons = protocol["comparisons"]
    repeats = protocol["new_repeat_indices"]
    index = {(r["graph"], r["repeat"], r["branch"]): r["tail"]["perceived_majority"] for r in rows}
    diffs = np.array([[[index[g, r, a] - index[g, r, b] for a, b in comparisons]
                       for r in repeats] for g in range(3)])
    rng = np.random.default_rng(protocol["analysis_seed"])
    estimate = diffs.mean(axis=(0, 1))
    draws = np.array([diffs[g, rng.integers(len(repeats), size=(protocol["bootstrap_repeats"], len(repeats)))]
                      for g in range(3)]).mean(axis=(0, 2))
    flat = diffs.reshape(-1, len(comparisons))
    simulations = 100_000
    signs = rng.integers(0, 2, size=(simulations, len(flat)), dtype=np.int8) * 2 - 1
    null = signs @ flat / len(flat)
    raw = (1 + (np.abs(null) >= np.abs(estimate) - 1e-12).sum(axis=0)) / (simulations + 1)
    order = np.argsort(raw)
    adjusted = np.empty_like(raw)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(raw[order] * np.arange(len(order), 0, -1)))
    intervals = np.quantile(draws, [0.025, 0.975], axis=0)
    return {
        "groups": len(flat), "sign_flips": simulations, "seed": protocol["analysis_seed"],
        "assumption": "Whole-group paired differences are sign-exchangeable under the null, conditional on the three networks.",
        "comparisons": [{"conditions": pair, "difference": estimate[i], "ci95": intervals[:, i].tolist(),
                         "p_raw": raw[i], "p_holm": adjusted[i]}
                        for i, pair in enumerate(comparisons)],
    }


def historical_comparison(rows, repo):
    """Describe the archived and new cohorts separately and together."""
    historical = read(repo / "experiments/2026-11-16/03_positions/results.json")
    historical_keys = {(row["graph"], row["repeat"], row["branch"]) for row in historical}
    expected = {(graph, repeat, branch) for graph in range(3) for repeat in range(3)
                for branch in ("TH", "TL", "U0")}
    require(len(historical) == len(expected) and historical_keys == expected, "Incomplete historical grid")
    hist = {branch: [row["summary"]["perceived_majority"] for row in historical if row["branch"] == branch]
            for branch in ("TH", "TL", "U0")}
    new = {branch: [row["tail"]["perceived_majority"] for row in rows if row["branch"] == branch]
           for branch in hist}
    return {
        "scope": "Descriptive comparison. The old cohort used multiple model routes; inference uses only the new cohort.",
        "historical_runs": len(historical),
        "new_runs": len(rows),
        "mean_q": {branch: {"historical": mean(hist[branch]), "new": mean(new[branch]),
                            "combined_descriptive": mean(hist[branch] + new[branch])}
                   for branch in hist},
    }


def audit(repo, prepared, runs, trial):
    protocol, tasks, groups, networks, _ = validate(repo, prepared)
    selected = [task for task in tasks if not trial or task["phase"] == "technical_trial"]
    missing = [t["task_id"] for t in selected if not (runs / t["task_id"] / "trajectory.jsonl.gz").is_file()]
    require(not missing, f"{len(missing)} incomplete trajectories; first: {', '.join(missing[:9])}")
    runtime = read(runs / "runtime.json")
    require(runtime["prepared_manifest_sha256"] == sha(prepared / "preparation_manifest.json")
            and runtime["runner_sha256"] == sha(repo / "scripts/run_exp1_extension.py")
            and runtime["temperature"] == 0 and runtime["thinking"] == "disabled"
            and runtime["timeout"] == 180 and runtime["max_tokens"] == 4096, "Runtime identity mismatch")
    rows = []
    for task in selected:
        graph, repeat, branch = task["graph"], task["repeat"], task["branch"]
        group = groups[graph, repeat]["data"]
        directory = runs / task["task_id"]
        require(read(directory / "input_identity.json") == task, f"Run identity mismatch: {directory}")
        result = read(directory / "result.json")
        net = read(repo / task["network"])["data"]["network"]
        advocates = set(group["positions"][branch])
        settings = protocol["settings"]
        require(result["settings"] == settings and result["branch"] == branch and result["size"] == 1000
                and result["k"] == 200 and result["tail_rounds"] == 3 and result["seed"] == task["group_seed"],
                f"Result condition mismatch: {directory}")
        expected = {key: runtime[key] for key in ("base_url", "model", "timeout", "max_tokens")}
        from majority.llm import prompt_hash
        calls = audit_trajectory(directory / "trajectory.jsonl.gz", {
            "network": net, "profiles": group["profiles"], "advocates": sorted(advocates),
            "settings": settings, "group_seed": task["group_seed"], "branch": branch,
            "observer_count": 50, "llm": expected, "thinking": "disabled", "prompt_sha256": prompt_hash("perception"),
        }, set(networks[graph].observers), advocates, result["metrics"])
        tail = {field: mean(row[field] for row in result["metrics"][-3:]) for field in FIELDS}
        require(all(abs(tail[field] - result["summary"][field]) < 1e-10 for field in FIELDS),
                f"Tail summary mismatch: {directory}")
        usage, models = audit_calls(directory / "calls.jsonl", calls, runtime["model"], task["task_id"])
        rows.append({"graph": graph, "repeat": repeat, "branch": branch, "tail": tail,
                     "api_usage": usage, "response_models": models})
    report = {"batch": "technical_trial" if trial else "supplement", "runtime": runtime,
              "trajectories": rows, "new_groups_inference": None}
    if not trial:
        require({model for row in rows for model in row["response_models"]} == {"deepseek-flash"},
                "Response model changed across the supplemental batch")
        report["new_groups_inference"] = paired_q(rows, protocol)
        report["historical_comparison"] = historical_comparison(rows, repo)
    return report


def write_report(report, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                       encoding="utf-8")
    rows = report["trajectories"]
    counts = Counter()
    for row in rows:
        counts.update(row["api_usage"])
    lines = ["# 主实验一补充验证" if report["new_groups_inference"] else "# 主实验一补充验证技术试跑",
             "", f"{len(rows)} 条十轮完整轨迹通过独立审计。Q 和 J 取第 8–10 轮均值。", "",
             "| 网络 | 参数组 | 位置 | Q | J |", "|---:|---:|---|---:|---:|"]
    for row in rows:
        tail = row["tail"]
        lines.append(f"| {row['graph'] + 1} | {row['repeat'] + 1} | {row['branch']} | "
                     f"{tail['perceived_majority']:.2%} | {tail['feed_majority']:.2%} |")
    if inference := report["new_groups_inference"]:
        comparisons = inference["comparisons"]
        lines += ["", "## 新增配对组的预定比较", "",
                  "| 比较 | 差（百分点） | 95% 区间 | Holm p |", "|---|---:|---:|---:|"]
        for row in comparisons:
            low, high = row["ci95"]
            lines.append(f"| {' − '.join(row['conditions'])} | {100 * row['difference']:+.2f} | "
                         f"[{100 * low:+.2f}, {100 * high:+.2f}] | {row['p_holm']:.4f} |")
        lines += ["", f"配对单位为新增的 {inference['groups']} 组，条件限定在原来的三张网络。"
                  "历史组和合并结果作描述性参照，详见 audit.json。"]
        plot_comparisons(comparisons, output / "paired_q.png")
        lines += ["", "![新增组配对比较](paired_q.png)"]
    else:
        lines += ["", "技术试跑用于完整性与实际用量审阅。"]
    lines += ["", "## 已知调用用量", "",
              f"实际请求 {counts['actual_calls']:,} 次，已知输入 {counts['prompt_tokens']:,} Token，"
              f"已知输出 {counts['completion_tokens']:,} Token。用量缺失情况见 audit.json。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_comparisons(comparisons, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 3), constrained_layout=True)
    for index, row in enumerate(comparisons):
        low, high = row["ci95"]
        ax.hlines(index, 100 * low, 100 * high, color="#30627e", linewidth=2)
        ax.scatter(100 * row["difference"], index, color="#164963")
    ax.set_yticks(range(len(comparisons)), [" - ".join(row["conditions"]) for row in comparisons])
    ax.invert_yaxis()
    ax.axvline(0, color="#596574", linewidth=1, linestyle="--")
    ax.set_xlabel("Q difference (percentage points), 95% bootstrap CI")
    ax.set_title("Main experiment 1 supplement | 21 matched groups")
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--trial", action="store_true")
    parser.add_argument("--output", type=Path, help="Write audit JSON and report to a new directory")
    args = parser.parse_args()
    report = audit(args.repo.resolve(), args.prepared.resolve(), args.runs.resolve(), args.trial)
    if args.output:
        write_report(report, args.output)
    else:
        print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
