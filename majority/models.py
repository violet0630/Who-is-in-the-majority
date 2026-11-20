"""冻结输入与不可变快照；可变历史由 Simulation 独占。"""

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from math import isfinite


class Stance(IntEnum):
    OPPOSE = -1
    SUPPORT = 1


class Action(str, Enum):
    AGREE = "agree"
    DISAGREE = "disagree"
    POST = "post"
    SILENT = "silent"


@dataclass(frozen=True)
class Network:
    communities: tuple[int, ...]
    following: tuple[frozenset[int], ...]
    observers: frozenset[int]
    fans: tuple[frozenset[int], ...] = field(init=False)

    def __post_init__(self) -> None:
        nodes = set(range(len(self.communities)))
        if len(self.following) != len(nodes) or not self.observers <= nodes:
            raise ValueError("网络节点、邻接表与观测者编号不一致")
        if not self.observers:
            raise ValueError("观测集合不能为空")
        fans: list[set[int]] = [set() for _ in nodes]
        for node, followed in enumerate(self.following):
            if not followed or node in followed or not followed <= nodes:
                raise ValueError("每个节点须关注有效的其他节点，且不允许自环")
            for target in followed:
                fans[target].add(node)
        object.__setattr__(self, "fans", tuple(frozenset(group) for group in fans))


@dataclass(frozen=True)
class Profile:
    baseline: float
    strength: float

    def __post_init__(self) -> None:
        if not 0 < self.baseline < 1 or self.strength not in (0.4, 0.8):
            raise ValueError("需要 0 < g0 < 1，h 为 0.4 或 0.8")


@dataclass(frozen=True)
class AgentState:
    stance: Stance
    perceived_support: float = 0.5
    feedback_value: float = 0.0


@dataclass(frozen=True)
class Coefficients:
    theta: tuple[float, float, float, float]
    kappa: float
    gamma: float
    beta: float = 10.0

    def __post_init__(self) -> None:
        values = (*self.theta, self.kappa, self.gamma, self.beta)
        if len(self.theta) != 4 or not all(isfinite(x) for x in values):
            raise ValueError("theta 需要四个有限系数")
        if min(self.kappa, self.gamma, self.beta) < 0:
            raise ValueError("beta、kappa、gamma 必须非负")


@dataclass(frozen=True)
class RunConfig:
    rounds: int = 20
    window: int = 2
    read_limit: int = 5
    alpha: float = 0.05
    rho: float = 1.0
    cross_fan_comments: bool = True
    advocate_stance: Stance = Stance.OPPOSE
    private_update: bool = False

    def __post_init__(self) -> None:
        if min(self.rounds, self.window, self.read_limit) < 1:
            raise ValueError("轮数、窗口和阅读上限必须为正")
        if not 0 <= self.alpha <= 1 or not isfinite(self.rho) or self.rho <= 0:
            raise ValueError("需要 0 <= alpha <= 1 且 rho > 0")
        if not isinstance(self.advocate_stance, Stance):
            raise ValueError("倡议者立场须为 Stance 枚举")


@dataclass(frozen=True)
class Text:
    id: str
    body: str


@dataclass(frozen=True)
class OriginalText:
    text: Text
    stance: Stance
    agree: tuple[Text, ...]
    disagree: tuple[Text, ...]


@dataclass(frozen=True)
class TextBank:
    originals: tuple[OriginalText, ...]

    def __post_init__(self) -> None:
        if {item.stance for item in self.originals} != set(Stance):
            raise ValueError("文本库需要双方原创候选")
        identifiers = []
        for item in self.originals:
            if not item.agree or not item.disagree:
                raise ValueError("每条原创需要双方评论候选")
            identifiers.extend(text.id for text in (item.text, *item.agree, *item.disagree))
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("候选文本编号必须全局唯一")


@dataclass(frozen=True)
class Message:
    id: str
    author: int
    round: int
    action: Action
    stance: Stance
    text: Text
    original_id: str | None
    audience: frozenset[int]


@dataclass(frozen=True)
class Reading:
    visible: tuple[str, ...] = ()
    read: tuple[str, ...] = ()
    context: tuple[str, ...] = ()
    targets: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionProbabilities:
    agree: float
    disagree: float
    post: float
    silent: float


@dataclass(frozen=True)
class Decision:
    action: Action
    probabilities: ActionProbabilities
    public_stance: int = 0
    target_id: str | None = None
