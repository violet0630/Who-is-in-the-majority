"""发布 t 轮消息，观察 t 轮信息，再统一提交 t+1 轮决策。"""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

from .behavior import decide, update_feedback
from .feed import make_message, read_feed
from .models import (
    Action, AgentState, Coefficients, Decision, Message, Network,
    Profile, Reading, RunConfig, Stance, TextBank,
)
from .perception import (
    Exposure, PerceptionModel, PerceptionRequest, PerceptionResult, ViewedMessage,
)
from .randomness import RandomStreams


@dataclass(frozen=True)
class NodeRecord:
    node: int
    before: AgentState
    after: AgentState
    reading: Reading
    notifications: tuple[str, ...]
    perception: PerceptionResult | None
    next_decision: Decision


@dataclass(frozen=True)
class RoundRecord:
    branch: str
    round: int
    published: tuple[Message, ...]
    nodes: tuple[NodeRecord, ...]
    advocate_stance: Stance


class Simulation:
    def __init__(
        self, network: Network, profiles: tuple[Profile, ...], advocates: frozenset[int],
        bank: TextBank, coefficients: Coefficients, config: RunConfig,
        perception: PerceptionModel, streams: RandomStreams, branch: str, workers: int = 1,
    ) -> None:
        nodes = set(range(len(network.communities)))
        if len(profiles) != len(nodes) or not advocates <= nodes:
            raise ValueError("个人参数或倡议者编号与网络不匹配")
        if advocates & network.observers:
            raise ValueError("观测者不能担任倡议者")
        if not 0 < len(advocates) < len(nodes) / 2:
            raise ValueError("主实验倡议者人数必须大于零且少于全网一半")
        if workers < 1:
            raise ValueError("workers must be positive")
        self.workers = workers
        self.network, self.profiles, self.advocates = network, profiles, advocates
        self.bank, self.coefficients, self.config = bank, coefficients, config
        self.perception, self.streams, self.branch = perception, streams, branch
        self.states = tuple(AgentState(config.advocate_stance if node in advocates
                                       else Stance(-config.advocate_stance))
                            for node in range(len(profiles)))
        self.messages: dict[str, Message] = {}
        self.history: list[list[Exposure]] = [[] for _ in profiles]
        self.next_round = 1
        self.initial_decisions = tuple(self._decide(node, 1, state, ())
                                       for node, state in enumerate(self.states))
        self.pending = self._prepare_messages(1, self.initial_decisions)

    def _decide(
        self, node: int, round: int, state: AgentState, targets: tuple[str, ...],
    ) -> Decision:
        return decide(
            node, round, self.profiles[node], state, node in self.advocates,
            tuple(self.messages[id] for id in targets), self.coefficients,
            self.config, self.streams,
        )

    def _prepare_messages(self, round: int, decisions: tuple[Decision, ...]) -> tuple[Message, ...]:
        return tuple(
            make_message(self.branch, node, round, decision, self.network, self.bank,
                         self.messages, self.config, self.streams.for_event(node, round, "text"))
            for node, decision in enumerate(decisions) if decision.action != Action.SILENT
        )

    def _view(self, ids: tuple[str, ...]) -> tuple[ViewedMessage, ...]:
        messages = (self.messages[id] for id in ids)
        return tuple(ViewedMessage(m.id, m.author, m.round, m.text.body, m.original_id)
                     for m in messages)

    def step(self) -> RoundRecord:
        if self.next_round > self.config.rounds:
            raise StopIteration("已完成设定轮次")
        round = self.next_round
        published = self.pending
        self.messages.update((message.id, message) for message in published)
        notifications: list[list[Message]] = [[] for _ in self.profiles]
        for message in published:
            if message.original_id is not None:
                author = self.messages[message.original_id].author
                notifications[author].append(message)

        readings = tuple(read_feed(node, round, self.messages, self.config,
                                   self.streams.for_event(node, round, "read"))
                         for node in range(len(self.profiles)))
        exposures = tuple(Exposure(round, self._view(r.read), self._view(r.context)) for r in readings)
        active = [node for node, reading in enumerate(readings) if reading.read]
        requests = [PerceptionRequest(node, self.profiles[node], self.states[node],
                                      tuple(self.history[node]), exposures[node]) for node in active]
        if self.workers == 1:
            results = dict(zip(active, map(self.perception.estimate, requests)))
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                results = dict(zip(active, pool.map(self.perception.estimate, requests)))
        records = []
        for node, before in enumerate(self.states):
            reading = readings[node]
            result = results.get(node)
            after = replace(before, perceived_support=result.support) if result else before
            replies = notifications[node]
            agree = sum(message.action == Action.AGREE for message in replies)
            after = update_feedback(after, agree, len(replies) - agree, self.config.alpha)
            decision = self._decide(node, round + 1, after, reading.targets)
            records.append(NodeRecord(node, before, after, reading,
                                      tuple(message.id for message in replies), result, decision))

        pending = self._prepare_messages(round + 1, tuple(row.next_decision for row in records))
        self.states = tuple(row.after for row in records)
        for history, exposure in zip(self.history, exposures):
            history.append(exposure)
        self.pending = pending
        self.next_round += 1
        return RoundRecord(self.branch, round, published, tuple(records), self.config.advocate_stance)

    def run(self) -> Iterator[RoundRecord]:
        while self.next_round <= self.config.rounds:
            yield self.step()
