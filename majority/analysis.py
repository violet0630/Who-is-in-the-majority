"""指标命名对应 PDF 第 4–5 页；沉默节点始终保留在观测集合分母中。"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import mean
from random import Random
from collections import defaultdict

from .models import Message, Network, Stance, message_order
from .simulation import RoundRecord


@dataclass(frozen=True)
class RoundMetrics:
    round: int
    true_support: float
    feed_majority: float
    perceived_majority: float
    perception_bias: float
    public_support: float
    opponent_silence: float | None
    public_conformity: float | None
    empty_readers: int
    advocate_stance: Stance
    true_advocate_support: float
    advocate_perception_bias: float
    public_advocate_stance: float


def measure(record: RoundRecord, network: Network, messages: Mapping[str, Message]) -> RoundMetrics:
    rows = [row for row in record.nodes if row.node in network.observers]
    focal = record.advocate_stance
    opponents = [row for row in rows if row.before.stance != focal]
    published = {message.author: message for message in sorted(record.published, key=message_order)}
    public_supporters = {node for node, m in published.items() if m.stance == Stance.SUPPORT}
    public_advocates = {node for node, m in published.items() if m.stance == focal}
    true_support = mean(row.before.stance == Stance.SUPPORT for row in record.nodes)
    focal_share = true_support if focal == Stance.SUPPORT else 1 - true_support
    perceived_focal = [row.after.perceived_support if focal == Stance.SUPPORT
                       else 1 - row.after.perceived_support for row in rows]
    feed_majorities = sum(
        sum(messages[id].stance == focal for id in row.reading.read)
        > len(row.reading.read) / 2 for row in rows
    )
    return RoundMetrics(
        record.round, true_support, feed_majorities / len(rows),
        mean(value > 0.5 for value in perceived_focal),
        mean(row.after.perceived_support - true_support for row in rows),
        len(public_supporters & network.observers) / len(rows),
        mean(row.node not in published for row in opponents) if opponents else None,
        mean(row.node in public_advocates for row in opponents) if opponents else None,
        sum(not row.reading.read for row in rows),
        focal, focal_share, mean(perceived_focal) - focal_share,
        len(public_advocates & network.observers) / len(rows),
    )


def tail_illusion(metrics: Sequence[RoundMetrics], last: int = 5) -> float:
    if last < 1 or len(metrics) < last:
        raise ValueError("完整运行必须包含足够的末尾轮次")
    if any(item.true_advocate_support >= 0.5 for item in metrics[-last:]):
        raise ValueError("当前汇总仅适用于倡议立场真实占比小于一半的主实验")
    return mean(item.perceived_majority for item in metrics[-last:])


@dataclass(frozen=True)
class RunSummary:
    network_id: str
    profile_id: str
    group_seed: int
    branch: str
    k: int
    rho: float
    value: float
    advocate_stance: Stance = Stance.OPPOSE


@dataclass(frozen=True)
class Comparison:
    branches: tuple[str, str]
    difference: float
    confidence_interval: tuple[float, float]
    adjusted_pvalue: float


def paired_inference(
    runs: Sequence[RunSummary], comparisons: Sequence[tuple[str, str]],
    *, repeats: int = 10000, seed: int = 42,
) -> tuple[Comparison, ...]:
    """同一 k/rho 条件：图内整组 bootstrap 10000 次、95% CI、Holm 校正。"""
    if repeats < 100 or not runs:
        raise ValueError("Provide runs and at least 100 resamples")
    if len({(r.k, r.rho, r.advocate_stance) for r in runs}) != 1:
        raise ValueError("Analyze one condition at a time")
    grouped = defaultdict(dict)
    for run in runs:
        key = (run.network_id, run.profile_id, run.group_seed)
        if run.branch in grouped[key]:
            raise ValueError("Duplicate branch in paired group")
        grouped[key][run.branch] = run.value
    needed = {branch for pair in comparisons for branch in pair}
    graphs = defaultdict(list)
    for (graph, _, _), values in grouped.items():
        if needed <= values.keys():
            graphs[graph].append(values)
    if not graphs or any(len(groups) < 2 for groups in graphs.values()):
        raise ValueError("Each graph requires at least two complete paired groups")
    rng = Random(seed)
    def contrasts(sample):
        return [mean(mean(group[a] - group[b] for group in groups) for groups in sample)
                for a, b in comparisons]
    observed = contrasts(list(graphs.values()))
    draws = [[] for _ in comparisons]
    for _ in range(repeats):
        sample = [[rng.choice(groups) for _ in groups] for groups in graphs.values()]
        for values, delta in zip(draws, contrasts(sample)):
            values.append(delta)
    pvalues = [(1 + sum(abs(x - estimate) >= abs(estimate) for x in values)) / (repeats + 1)
               for estimate, values in zip(observed, draws)]
    adjusted = [0.0] * len(pvalues)
    previous = 0.0
    for rank, index in enumerate(sorted(range(len(pvalues)), key=pvalues.__getitem__)):
        previous = max(previous, min(1.0, pvalues[index] * (len(pvalues) - rank)))
        adjusted[index] = previous
    result = []
    for index, (pair, estimate, values) in enumerate(zip(comparisons, observed, draws)):
        values.sort()
        interval = (values[int(0.025 * (repeats - 1))], values[int(0.975 * (repeats - 1))])
        result.append(Comparison(pair, estimate, interval, adjusted[index]))
    return tuple(result)

def grid_threshold(runs: Sequence[RunSummary], k_grid: Sequence[int], *, population: int = 100,
                   coverage: float = 0.8) -> float | None:
    """同一分支和 rho 下按图等权求网格阈值。"""
    if not runs or len({(r.branch, r.rho, r.advocate_stance) for r in runs}) != 1:
        raise ValueError("Provide one branch and one rho")
    if population < 1 or not 0 < coverage <= 1:
        raise ValueError("Invalid population or coverage")
    groups = defaultdict(lambda: defaultdict(list))
    for run in runs:
        groups[run.k][run.network_id].append(run.value)
    graph_ids = {run.network_id for run in runs}
    if any(set(groups[k]) != graph_ids for k in k_grid):
        raise ValueError("Each requested k must include all networks")
    reached = [k for k in sorted(set(k_grid))
               if mean(mean(values) for values in groups[k].values()) >= coverage]
    return reached[0] / population if reached else None
