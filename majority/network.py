"""有向度修正社团网络；i→j 表示 i 关注 j。"""

from dataclasses import dataclass
from math import expm1, isfinite
from random import Random

from .models import Network


@dataclass(frozen=True)
class NetworkConfig:
    community_sizes: tuple[int, ...] = (15, 20, 25, 40)
    s_out: float = 0.35
    s_in: float = 1.0
    between_weight: float = 0.2
    expected_edges: int = 800
    edge_range: tuple[int, int] = (760, 840)
    observer_count: int = 30
    max_attempts: int = 10000

    def __post_init__(self) -> None:
        n = sum(self.community_sizes)
        if not self.community_sizes or min(self.community_sizes) < 1 or n < 2:
            raise ValueError("社团人数须为正，网络至少有两个节点")
        if not all(isfinite(x) and x >= 0 for x in (self.s_out, self.s_in, self.between_weight)):
            raise ValueError("异质性和跨社团权重须为有限非负数")
        if len(self.community_sizes) > 1 and self.between_weight == 0:
            raise ValueError("多个社团在跨社团权重为 0 时无法弱连通")
        lower, upper = self.edge_range
        if not n <= lower <= self.expected_edges <= upper <= n * (n - 1):
            raise ValueError("边数范围须容纳期望边数，且满足最小出度与简单图约束")
        if self.expected_edges == n * (n - 1):
            raise ValueError("有限 lambda 无法生成期望完全图")
        if not 1 <= self.observer_count < n or self.max_attempts < 1:
            raise ValueError("观测者数量或重采样次数无效")


def _connected(following: tuple[frozenset[int], ...]) -> bool:
    neighbors = [set(group) for group in following]
    for node, targets in enumerate(following):
        for target in targets:
            neighbors[target].add(node)
    seen, pending = {0}, [0]
    while pending:
        for node in neighbors[pending.pop()] - seen:
            seen.add(node)
            pending.append(node)
    return len(seen) == len(following)


def observer_strata(communities: tuple[int, ...], fan_counts: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """每社团按 (粉丝数, 节点编号) 排序，分成三个秩分层。"""
    strata = []
    for community in sorted(set(communities)):
        nodes = sorted((i for i, c in enumerate(communities) if c == community),
                       key=lambda i: (fan_counts[i], i))
        for tier in range(3):
            group = tuple(node for rank, node in enumerate(nodes) if rank * 3 // len(nodes) == tier)
            if group:
                strata.append(group)
    return tuple(strata)


def _observers(communities: tuple[int, ...], following: tuple[frozenset[int], ...], count: int,
               rng: Random) -> frozenset[int]:
    n = len(communities)
    fans = tuple(sum(node in targets for targets in following) for node in range(n))
    strata = observer_strata(communities, fans)
    quotas = [count * len(group) // n for group in strata]
    order = sorted(range(len(strata)), key=lambda i: (-(count * len(strata[i]) % n), i))
    for index in order[:count - sum(quotas)]:
        quotas[index] += 1
    return frozenset(node for group, quota in zip(strata, quotas) for node in rng.sample(group, quota))


def generate_network(config: NetworkConfig, rng: Random) -> Network:
    labels = [c for c, size in enumerate(config.community_sizes) for _ in range(size)]
    rng.shuffle(labels)
    communities = tuple(labels)
    n = len(labels)
    out_weights = [rng.lognormvariate(-config.s_out ** 2 / 2, config.s_out) for _ in labels]
    in_weights = [rng.lognormvariate(-config.s_in ** 2 / 2, config.s_in) for _ in labels]
    pairs = [(i, j, out_weights[i] * in_weights[j]
              * (1 if labels[i] == labels[j] else config.between_weight))
             for i in range(n) for j in range(n) if i != j]
    low, high = 0.0, 1.0
    while sum(-expm1(-high * weight) for _, _, weight in pairs) < config.expected_edges:
        high *= 2
    for _ in range(60):
        middle = (low + high) / 2
        if sum(-expm1(-middle * weight) for _, _, weight in pairs) < config.expected_edges:
            low = middle
        else:
            high = middle
    probabilities = [(i, j, -expm1(-(low + high) / 2 * weight)) for i, j, weight in pairs]
    for _ in range(config.max_attempts):
        adjacency: list[set[int]] = [set() for _ in labels]
        for i, j, probability in probabilities:
            if rng.random() < probability:
                adjacency[i].add(j)
        edges = sum(map(len, adjacency))
        if not config.edge_range[0] <= edges <= config.edge_range[1] or not all(adjacency):
            continue
        following = tuple(frozenset(group) for group in adjacency)
        if _connected(following):
            observers = _observers(communities, following, config.observer_count, rng)
            return Network(communities, following, observers)
    raise RuntimeError(f"固定节点权重下 {config.max_attempts} 次重抽边均不满足约束")


@dataclass(frozen=True)
class NetworkStatistics:
    following_counts: tuple[int, ...]
    fan_counts: tuple[int, ...]
    edges: int
    cross_community_fraction: float
    reciprocal_edge_fraction: float
    undirected_clustering: float
    observer_strata: tuple[tuple[int, ...], ...]


def network_statistics(network: Network) -> NetworkStatistics:
    edges = [(i, j) for i, targets in enumerate(network.following) for j in targets]
    neighbors = [network.following[i] | network.fans[i] for i in range(len(network.communities))]
    clustering = []
    for group in neighbors:
        links = sum(len(neighbors[node] & group) for node in group)
        clustering.append(links / (len(group) * (len(group) - 1)) if len(group) > 1 else 0)
    fan_counts = tuple(map(len, network.fans))
    return NetworkStatistics(
        tuple(map(len, network.following)), fan_counts, len(edges),
        sum(network.communities[i] != network.communities[j] for i, j in edges) / len(edges),
        sum(i in network.following[j] for i, j in edges) / len(edges),
        sum(clustering) / len(clustering), observer_strata(network.communities, fan_counts),
    )
