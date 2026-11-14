import argparse
import json
from pathlib import Path

from majority.analysis import measure, tail_illusion
from majority.experiments import SeedSelection, plan_group
from majority.models import Coefficients, Network, OriginalText, Profile, RunConfig, Stance, Text, TextBank
from majority.perception import PerceptionRequest, PerceptionResult
from majority.randomness import RandomStreams
from majority.simulation import Simulation
from majority.storage import write_record


class DemoPerception:
    def estimate(self, request: PerceptionRequest) -> PerceptionResult:
        support = sum("[支持]" in message.text for message in request.current.messages)
        value = support / len(request.current.messages)
        return PerceptionResult(value, json.dumps({"demo_only": True, "support": value}))


def demo_inputs() -> tuple[Network, tuple[Profile, ...], TextBank]:
    network = Network(
        communities=(0, 0, 0, 0, 1, 1, 1, 1),
        following=tuple(frozenset({(i + 1) % 8, (i + 3) % 8}) for i in range(8)),
        observers=frozenset({4, 5, 6, 7}),
    )
    profiles = tuple(Profile(0.65, 0.4 if i % 2 == 0 else 0.8) for i in range(8))
    bank = TextBank(tuple(
        OriginalText(
            Text(f"post-{stance.value}", f"[{label}] 这是一条用于演示的原创帖。"), stance,
            (Text(f"agree-{stance.value}", f"[{label}] 我赞同这条原帖。"),),
            (Text(f"disagree-{stance.value}", f"[{opposite}] 我反对这条原帖。"),),
        ) for stance, label, opposite in (
            (Stance.SUPPORT, "支持", "反对"), (Stance.OPPOSE, "反对", "支持"),
        )
    ))
    return network, profiles, bank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/demo.jsonl"))
    args = parser.parse_args()
    network, profiles, bank = demo_inputs()
    coefficients = Coefficients(theta=(0, 0, 1, 0), kappa=1, gamma=1)
    config = RunConfig(rho=3, advocate_stance=Stance.SUPPORT)
    # 演示不伪称经过高低种子优化，只运行同种子集的两条随机分支。
    selection = SeedSelection(None, None, frozenset({0}), 0, None)
    specs = plan_group("demo-network", "demo-profiles", selection, 20261114)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        write_record(file, "manifest", {
            "demo_only": True, "model": "demo-text-marker", "prompt_version": "none",
            "network": network, "profiles": profiles, "bank": bank,
            "coefficients": coefficients, "config": config, "seeds": selection,
        })
        for spec in specs:
            simulation = Simulation(
                network, profiles, spec.advocates, bank, coefficients, config,
                DemoPerception(), RandomStreams(spec.group_seed, spec.stream), spec.branch,
            )
            write_record(file, "run", {"spec": spec, "initial_decisions": simulation.initial_decisions})
            metrics = []
            for record in simulation.run():
                write_record(file, "round", record)
                metrics.append(measure(record, network, simulation.messages))
                write_record(file, "metrics", metrics[-1])
            print(f"{spec.branch}: demo tail perception majority = {tail_illusion(metrics):.3f}")
    print(f"Demo only. Log: {args.output.resolve()}")


if __name__ == "__main__":
    main()
