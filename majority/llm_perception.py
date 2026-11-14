"""感知提示仅暴露个人已见信息，隐藏实验分支、真实支持率与受众列表。"""

from dataclasses import dataclass

from .llm import JsonChat, LLMValidationError, prompt_hash
from .perception import Exposure, PerceptionRequest, PerceptionResult, ViewedMessage


def perception_payload(topic: str, request: PerceptionRequest, scenario: str = "") -> dict:
    exposures = (*request.history, request.current)
    aliases = {m.id: f"r{m.round}-u{m.author}"
               for exposure in exposures for m in (*exposure.messages, *exposure.context)}

    def exposure_data(exposure: Exposure) -> dict:
        def message_data(message: ViewedMessage) -> dict:
            return {"id": aliases[message.id], "author": message.author, "round": message.round,
                    "text": message.text,
                    "original_id": aliases[message.original_id] if message.original_id else None}
        return {"round": exposure.round, "messages": [message_data(m) for m in exposure.messages],
                "context": [message_data(m) for m in exposure.context]}

    return {
        "topic": topic, "scenario": scenario,
        "self": {"node": request.node, "private_stance": int(request.state.stance),
                 "strength": request.profile.strength,
                 "previous_support": request.state.perceived_support},
        "history": [exposure_data(exposure) for exposure in request.history],
        "current": exposure_data(request.current),
    }


@dataclass
class LLMPerception:
    chat: JsonChat
    topic: str
    run_id: str
    scenario: str = ""

    def estimate(self, request: PerceptionRequest) -> PerceptionResult:
        if not request.current.messages:
            raise ValueError("空阅读应由 Simulation 沿用旧判断，不调用 LLM")
        payload = perception_payload(self.topic, request, self.scenario)
        evidence = {message["id"] for exposure in (*payload["history"], payload["current"])
                    for message in (*exposure["messages"], *exposure["context"])}

        def parse(data: dict) -> tuple[float, str, tuple[str, ...]]:
            if set(data) != {"support", "reason", "evidence_ids"}:
                raise ValueError("感知输出字段不完整或存在额外字段")
            support, reason, ids = data["support"], data["reason"], data["evidence_ids"]
            if type(support) not in (int, float) or not 0 <= support <= 1:
                raise ValueError("support 必须是 [0,1] 内数值")
            if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 120:
                raise ValueError("reason 必须为 1–120 字")
            if not isinstance(ids, list) or len(ids) > 5 or any(not isinstance(id, str) for id in ids):
                raise ValueError("evidence_ids 必须是最多五个字符串")
            if len(set(ids)) != len(ids) or not set(ids) <= evidence:
                raise ValueError("证据引用重复或不在个人阅读历史中")
            return float(support), reason, tuple(ids)

        context = {"run_id": self.run_id, "node": request.node, "round": request.current.round}
        try:
            parsed, result = self.chat.request("perception", payload, parse, temperature=0, context=context)
        except LLMValidationError:
            corrected = {**payload, "allowed_evidence_ids": sorted(evidence),
                         "validation_feedback": "请重新输出规定的三个字段。证据必须来自 allowed_evidence_ids，最多五条且不重复。support 为0至1数值，reason为1至120字。"}
            parsed, result = self.chat.request("perception", corrected, parse, temperature=0,
                                               context={**context, "correction": 1})
        support, reason, ids = parsed
        return PerceptionResult(
            support, result.raw_json, result.input_tokens, result.output_tokens,
            reason, ids, result.request_id, result.model, prompt_hash("perception"),
            result.elapsed_seconds,
        )
