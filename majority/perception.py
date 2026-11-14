"""LLM 仅估计社区支持率；行为由 behavior 模块抽样。"""

from dataclasses import dataclass
from typing import Protocol

from .models import AgentState, Profile


@dataclass(frozen=True)
class ViewedMessage:
    id: str
    author: int
    round: int
    text: str
    original_id: str | None


@dataclass(frozen=True)
class Exposure:
    round: int
    messages: tuple[ViewedMessage, ...]
    context: tuple[ViewedMessage, ...]


@dataclass(frozen=True)
class PerceptionRequest:
    node: int
    profile: Profile
    state: AgentState
    history: tuple[Exposure, ...]
    current: Exposure


@dataclass(frozen=True)
class PerceptionResult:
    support: float
    raw_json: str
    input_tokens: int | None = 0
    output_tokens: int | None = 0
    reason: str = ""
    evidence_ids: tuple[str, ...] = ()
    request_id: str = ""
    model: str = ""
    prompt_sha256: str = ""
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not 0 <= self.support <= 1:
            raise ValueError("LLM 支持率必须在 [0, 1]，不静默截断无效结果")
        if any(value is not None and value < 0 for value in (self.input_tokens, self.output_tokens)):
            raise ValueError("token 用量不能为负")


class PerceptionModel(Protocol):
    def estimate(self, request: PerceptionRequest) -> PerceptionResult:
        """接入冻结模型/提示/温度；解析并校验 JSON，保留原始响应及用量。"""
        ...
