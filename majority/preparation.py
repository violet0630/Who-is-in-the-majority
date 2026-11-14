"""研究准备阶段。"""

from collections.abc import Sequence
from dataclasses import dataclass
from random import Random

from .experiments import SeedSelection
from .models import Action, Coefficients, Network, Profile, Stance
from .network import NetworkConfig, generate_network


@dataclass(frozen=True)
class SearchConfig:
    tolerance: float = 0.05
    starts: int = 100
    swaps: int = 1000
    retry_starts: int = 1000


@dataclass(frozen=True)
class ExperimentPlan:
    networks: int = 3
    profile_groups: int = 10
    k_grid: tuple[int, ...] = (2, 4, 6, 8, 12, 16, 20)
    rho_grid: tuple[float, ...] = (1, 3, 6)
    coverage: float = 0.8
    tail_rounds: int = 5
    screening_repeats: int = 100


@dataclass(frozen=True)
class PilotRecord:
    trajectory_id: str
    profile: Profile
    stance: Stance
    perceived_support: float
    updated_feedback: float
    has_targets: bool
    action: Action
    target_stance: Stance | None


@dataclass(frozen=True)
class SeedDiagnostics:
    illusion: float
    fan_total: int
    fan_union: int
    baseline_total: float


def seed_diagnostics(
    network: Network, profiles: tuple[Profile, ...], seeds: frozenset[int],
) -> SeedDiagnostics:
    ordinary = set(range(len(profiles))) - seeds
    illusion = sum(len(network.following[node] & seeds) > len(network.following[node]) / 2
                   for node in ordinary) / len(ordinary)
    return SeedDiagnostics(
        illusion, sum(len(network.fans[node]) for node in seeds),
        len(frozenset().union(*(network.fans[node] for node in seeds))),
        sum(profiles[node].baseline for node in seeds),
    )


def draw_profiles(size: int, rng: Random) -> tuple[Profile, ...]:
    if size < 2 or size % 2:
        raise ValueError("两档立场强度各半，需要正偶数节点")
    strengths = [0.4] * (size // 2) + [0.8] * (size // 2)
    rng.shuffle(strengths)
    return tuple(Profile(rng.betavariate(0.24, 0.85), h) for h in strengths)


def calibrate(records: Sequence[PilotRecord]) -> Coefficients:
    """普通节点的独立预试轨迹；偏置、约束、L2 网格及样本筛选见 DESIGN.md。"""
    raise NotImplementedError("按 trajectory_id 分割训练/验证集并冻结拟合结果")


def search_seeds(
    network: Network, profiles: tuple[Profile, ...], k: int,
    config: SearchConfig, rng: Random,
) -> SeedSelection:
    """同社团交换；匹配粉丝总数与 baseline 总和，排除 observers，记录实际 IG 差。"""
    groups = [sorted(i for i, label in enumerate(network.communities)
                     if label == community and i not in network.observers)
              for community in sorted(set(network.communities))]
    available = sum(map(len, groups))
    if not 0 < k < len(profiles) / 2 or k > available:
        raise ValueError("Invalid minority size")
    quotas = [k * len(group) // available for group in groups]
    order = sorted(range(len(groups)), key=lambda i: (-(k * len(groups[i]) % available), i))
    for i in order[:k - sum(quotas)]:
        quotas[i] += 1

    def draw():
        return frozenset(node for group, quota in zip(groups, quotas) for node in rng.sample(group, quota))

    def matches(a, b):
        for x, y in ((a.fan_total, b.fan_total), (a.baseline_total, b.baseline_total)):
            if x + y and 2 * abs(x - y) / (x + y) > config.tolerance:
                return False
        return True

    reference = draw()
    baseline = seed_diagnostics(network, profiles, reference)
    high = low = reference
    high_score = low_score = baseline.illusion
    attempts = 0
    mutable = [i for i, group in enumerate(groups) if 0 < quotas[i] < len(group)]
    for start in range(config.retry_starts):
        current = draw()
        for _ in range(config.swaps):
            attempts += 1
            score = seed_diagnostics(network, profiles, current)
            if matches(score, baseline):
                if score.illusion > high_score and matches(score, seed_diagnostics(network, profiles, low)):
                    high, high_score = current, score.illusion
                if score.illusion < low_score and matches(score, seed_diagnostics(network, profiles, high)):
                    low, low_score = current, score.illusion
            if not mutable:
                break
            group = groups[rng.choice(mutable)]
            outgoing = rng.choice(sorted(current.intersection(group)))
            incoming = rng.choice(sorted(set(group) - current))
            current = frozenset((current - {outgoing}) | {incoming})
        if start + 1 >= config.starts and high_score > low_score:
            break
    if high_score == low_score:
        return SeedSelection(None, None, reference, attempts, None)
    return SeedSelection(high, low, reference, attempts, high_score - low_score)
