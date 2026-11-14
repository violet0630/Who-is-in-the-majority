"""只组织已准备好的输入，不在运行中重新抽网络、个人参数或种子配置。"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from .models import Coefficients, Network, Profile, RunConfig, TextBank
from .perception import PerceptionModel
from .randomness import RandomStreams
from .simulation import RoundRecord, Simulation


@dataclass(frozen=True)
class SeedSelection:
    high: frozenset[int] | None
    low: frozenset[int] | None
    random: frozenset[int]
    attempts: int
    illusion_gap: float | None

    def __post_init__(self) -> None:
        if (self.high is None) != (self.low is None):
            raise ValueError("高低种子必须同时存在或同时缺失")
        if self.high is not None and not len(self.high) == len(self.low) == len(self.random):
            raise ValueError("同组种子人数必须相同")


@dataclass(frozen=True)
class RunSpec:
    network_id: str
    profile_id: str
    branch: str
    advocates: frozenset[int]
    group_seed: int
    stream: str


def plan_group(
    network_id: str, profile_id: str, seeds: SeedSelection, group_seed: int,
) -> tuple[RunSpec, ...]:
    branches = (("TH", seeds.high), ("TL", seeds.low),
                ("U0", seeds.random), ("U1", seeds.random))
    return tuple(
        RunSpec(network_id, profile_id, branch, advocates, group_seed,
                "independent" if branch == "U1" else "paired")
        for branch, advocates in branches if advocates is not None
    )


def run_group(
    specs: tuple[RunSpec, ...], network: Network, profiles: tuple[Profile, ...],
    bank: TextBank, coefficients: Coefficients, config: RunConfig,
    perception_factory: Callable[[], PerceptionModel],
) -> Iterator[tuple[RunSpec, tuple[RoundRecord, ...]]]:
    for spec in specs:
        simulation = Simulation(
            network, profiles, spec.advocates, bank, coefficients, config,
            perception_factory(), RandomStreams(spec.group_seed, spec.stream), spec.branch,
        )
        yield spec, tuple(simulation.run())
