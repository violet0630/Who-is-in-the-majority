"""python -m majority：生成网络/候选池、检查接口、运行单分支或汇总调用。"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from random import Random

from .analysis import measure
from .artifacts import load_bank, load_coefficients, load_network, save_artifact
from .judgments import summarize_calls, summarize_judgments
from .llm import JsonChat, LLMError, load_prompt, load_settings, open_client, prompt_hash
from .llm_perception import LLMPerception
from .models import AgentState, Profile, RunConfig, Stance
from .network import NetworkConfig, generate_network, network_statistics
from .perception import Exposure, PerceptionRequest, ViewedMessage
from .preparation import draw_profiles
from .randomness import RandomStreams
from .simulation import Simulation
from .storage import write_record
from .text_generation import generate_text_bank


def _network(args) -> None:
    if args.count < 1:
        raise ValueError("网络数量必须为正")
    config = NetworkConfig()
    for index in range(args.count):
        seed = args.seed + index
        network = generate_network(config, Random(seed))
        stats = network_statistics(network)
        save_artifact(args.output / f"network-{index}.json", "network", {
            "seed": seed, "config": config, "network": network, "statistics": stats,
        })
        print(f"network-{index}: {stats.edges} edges, {len(network.observers)} observers", flush=True)


def _texts(args, chat: JsonChat) -> None:
    study = json.loads(args.study.read_text(encoding="utf-8"))
    bank = generate_text_bank(study["topic"], chat, per_side=args.per_side,
                              comments_per_side=args.comments, scenario=study["scenario"])
    digest = hashlib.sha256(json.dumps(asdict(bank), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    save_artifact(args.output / "bank.json", "text_bank", {
        "study": study, "bank": bank, "sha256": digest, "model": chat.config.model,
        "prompt_hashes": {name: prompt_hash(name) for name in ("originals", "comments")},
        "temperature": 0.7, "semantic_review": "pending", "min_chars": 40, "max_chars": 80,
    })
    lines = ["# 候选文本池", "", study["topic"], "", study["scenario"], ""]
    for original in bank.originals:
        label = "支持当选" if original.stance == Stance.SUPPORT else "反对当选"
        lines.extend([f"## {original.text.id} · {label}", "", original.text.body, ""])
        for relation, texts in (("赞同原帖", original.agree), ("反对原帖", original.disagree)):
            lines.extend([f"### {relation}", ""])
            lines.extend(f"- {text.id}：{text.body}" for text in texts)
            lines.append("")
    (args.output / "candidate_pool.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {len(bank.originals)} originals and {len(bank.originals) * args.comments * 2} comments.")


def _check(args, chat: JsonChat) -> None:
    study, bank = load_bank(args.bank)
    messages = tuple(ViewedMessage(f"check:1:{i}", i, 1, item.text.body, None)
                     for i, item in enumerate((bank.originals[0], bank.originals[-1])))
    request = PerceptionRequest(99, Profile(0.2, 0.8), AgentState(Stance.SUPPORT), (),
                                Exposure(1, messages, ()))
    result = LLMPerception(chat, study["topic"], "check", study["scenario"]).estimate(request)
    save_artifact(args.output / "perception.json", "perception_check", result)
    print(f"API check: support={result.support:.3f}, model={result.model}")


def _run(args, chat: JsonChat) -> None:
    network = load_network(args.network)
    study, bank = load_bank(args.bank)
    coefficients = load_coefficients(args.coefficients)
    profiles = draw_profiles(len(network.communities), Random(args.seed))
    advocates = frozenset(int(node.strip()) for node in args.advocates.split(","))
    config = RunConfig(rounds=args.rounds, rho=args.rho, alpha=args.alpha, advocate_stance=Stance(study["advocate_stance"]))
    simulation = Simulation(
        network, profiles, advocates, bank, coefficients, config,
        LLMPerception(chat, study["topic"], args.branch, study["scenario"]),
        RandomStreams(args.seed, "independent" if args.branch == "U1" else "paired"), args.branch, workers=args.workers,
    )
    with (args.output / "trajectory.jsonl").open("x", encoding="utf-8") as file:
        write_record(file, "manifest", {
            "study": study, "network": network, "profiles": profiles, "advocates": advocates,
            "config": config, "coefficients": coefficients, "llm": chat.config,
            "seed": args.seed, "branch": args.branch, "bank": bank,
            "prompt_sha256": prompt_hash("perception"), "temperature": 0,
            "initial_decisions": simulation.initial_decisions,
        })
        for record in simulation.run():
            write_record(file, "round", record)
            write_record(file, "metrics", measure(record, network, simulation.messages))
            for observed_only in (True, False):
                write_record(file, "judgments", summarize_judgments(record, network, observed_only=observed_only))
            file.flush()
            print(f"{args.branch}: round {record.round}/{args.rounds}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-calls", type=Path)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    commands = parser.add_subparsers(dest="command", required=True)
    network = commands.add_parser("network", help="生成并保存正式网络")
    network.add_argument("--count", type=int, default=3)
    network.add_argument("--seed", type=int, default=20261114)
    network.add_argument("--output", type=Path, default=Path("runs/networks"))
    texts = commands.add_parser("texts", help="调用 LLM 生成候选池")
    texts.add_argument("--study", type=Path, default=Path("configs/study.json"))
    texts.add_argument("--per-side", type=int, default=5)
    texts.add_argument("--comments", type=int, default=2)
    texts.add_argument("--output", type=Path, default=Path("runs/candidate_pool"))
    check = commands.add_parser("check-llm", help="使用候选池检查一次真实感知调用")
    check.add_argument("--bank", type=Path, default=Path("runs/candidate_pool_final/bank.json"))
    check.add_argument("--output", type=Path, default=Path("runs/api_check"))
    run = commands.add_parser("run", help="运行给定输入的单分支；系数须显式提供")
    run.add_argument("--network", type=Path, required=True)
    run.add_argument("--bank", type=Path, required=True)
    run.add_argument("--coefficients", type=Path, required=True)
    run.add_argument("--advocates", required=True)
    run.add_argument("--branch", choices=("TH", "TL", "U0", "U1"), default="U0")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--rounds", type=int, default=20)
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--alpha", type=float, default=0.05)
    run.add_argument("--rho", type=float, default=3)
    run.add_argument("--output", type=Path, required=True)
    summary = commands.add_parser("summarize", help="重新汇总调用日志，包括失败请求")
    summary.add_argument("log", type=Path)
    args = parser.parse_args()
    if args.command == "network":
        _network(args)
        return
    if args.command == "summarize":
        print(json.dumps(summarize_calls(args.log), ensure_ascii=False, indent=2))
        return
    config, api_key = load_settings(args.env)
    args.output.mkdir(parents=True, exist_ok=False)
    save_artifact(args.output / "prompts.json", "prompts", {
        name: {"sha256": prompt_hash(name), "text": load_prompt(name)}
        for name in ("originals", "comments", "perception")
    })
    audit_path = args.output / "calls.jsonl"
    try:
        with open_client(config, api_key) as client, audit_path.open("x", encoding="utf-8") as audit:
            chat = JsonChat(client, config, audit, args.reuse_calls)
            {"texts": _texts, "check-llm": _check, "run": _run}[args.command](args, chat)
    except LLMError as exc:
        parser.exit(1, f"{exc}\n")
    finally:
        if audit_path.exists():
            save_artifact(args.output / "calls.summary.json", "call_summary", summarize_calls(audit_path))


if __name__ == "__main__":
    main()
