"""主实验的关系信息池；通知由调度器处理，不混入阅读抽样。"""

from collections.abc import Mapping
from random import Random

from .models import Action, Decision, Message, Network, Reading, RunConfig, Stance, TextBank


def read_feed(
    node: int, round: int, messages: Mapping[str, Message], config: RunConfig, rng: Random,
) -> Reading:
    earliest = round - config.window + 1
    visible = tuple(sorted(
        message.id for message in messages.values()
        if earliest <= message.round <= round and node in message.audience
    ))
    selected = tuple(sorted(rng.sample(visible, min(config.read_limit, len(visible)))))
    context = tuple(sorted({
        messages[id].original_id for id in selected if messages[id].original_id is not None
    }))
    originals = {id for id in selected if messages[id].action == Action.POST} | set(context)
    targets = tuple(sorted(
        id for id in originals
        if messages[id].author != node and earliest <= messages[id].round <= round
    ))
    return Reading(visible, selected, context, targets)


def make_message(
    branch: str, node: int, round: int, decision: Decision, network: Network,
    bank: TextBank, messages: Mapping[str, Message], config: RunConfig, rng: Random, slot: int = 0,
) -> Message:
    audience = network.fans[node]
    if decision.action == Action.POST:
        candidates = tuple(item.text for item in bank.originals
                           if item.stance == decision.public_stance)
    elif decision.action in (Action.AGREE, Action.DISAGREE):
        original = messages[decision.target_id]
        template = next(item for item in bank.originals if item.text.id == original.text.id)
        candidates = template.agree if decision.action == Action.AGREE else template.disagree
        if config.cross_fan_comments:
            audience = audience | network.fans[original.author]
    else:
        raise ValueError("沉默不产生消息")
    text = rng.choice(sorted(candidates, key=lambda item: item.id))
    message_id = f"{branch}:{round}:{node}" if slot == 0 else f"{branch}:{round}:{node}:c{slot}"
    return Message(
        message_id, node, round, decision.action,
        Stance(decision.public_stance), text, decision.target_id, audience - {node},
    )
