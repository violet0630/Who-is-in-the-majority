"""PDF 第 2–3 页公式；评论赞同/反对始终相对于目标原帖。"""

from dataclasses import replace
from math import exp, log, log1p

from .models import (
    Action, ActionProbabilities, AgentState, Coefficients, CommentDecision, Decision,
    Message, Profile, RunConfig,
)
from .randomness import RandomStreams


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + exp(-value))
    scaled = exp(value)
    return scaled / (1 + scaled)


def update_feedback(state: AgentState, agree: int, disagree: int, alpha: float) -> AgentState:
    reward = (agree - disagree) / (1 + agree + disagree)
    value = (1 - alpha) * state.feedback_value + alpha * reward
    return replace(state, feedback_value=value)


def decide(
    node: int,
    round: int,
    profile: Profile,
    state: AgentState,
    advocate: bool,
    targets: tuple[Message, ...],
    coefficients: Coefficients,
    config: RunConfig,
    streams: RandomStreams,
) -> Decision:
    baseline_logit = log(profile.baseline) - log1p(-profile.baseline)
    g = sigmoid(baseline_logit + coefficients.beta * state.feedback_value
                + (log(config.rho) if advocate else 0))
    headwind = (1 - state.stance * (2 * state.perceived_support - 1)) / 2
    theta0, theta_h, theta_d, theta_hd = coefficients.theta
    h = profile.strength
    c = sigmoid(theta0 + theta_h * h + theta_d * headwind + theta_hd * h * headwind)
    c = c if targets else 0.0
    field = coefficients.kappa * state.stance * h
    field += coefficients.gamma * (2 * state.perceived_support - 1)
    agreement = tuple(
        (1 + config.advocate_stance * target.stance) / 2 if advocate else sigmoid(2 * target.stance * field)
        for target in targets
    )
    mean_agreement = sum(agreement) / len(agreement) if agreement else 0.0
    probabilities = ActionProbabilities(
        g * c * mean_agreement, g * c * (1 - mean_agreement), g * (1 - c), 1 - g,
    )
    if streams.for_event(node, round, "speak").random() >= g:
        return Decision(Action.SILENT, probabilities)
    if streams.for_event(node, round, "comment").random() >= c:
        return Decision(Action.POST, probabilities, int(state.stance))
    count_rng = streams.for_event(node, round, "comment_count")
    share = count_rng.betavariate(config.comment_alpha, config.comment_beta)
    count = 1 + sum(count_rng.random() < share for _ in range(config.max_comments - 1))
    count = min(count, len(targets))
    target_rng = streams.for_event(node, round, "comment_targets")
    indexes = target_rng.sample(range(len(targets)), count)
    comments = []
    for slot, index in enumerate(indexes):
        target = targets[index]
        agrees = streams.for_event(node, round, f"stance:{slot}").random() < agreement[index]
        action = Action.AGREE if agrees else Action.DISAGREE
        stance = int(target.stance) if agrees else -int(target.stance)
        comments.append(CommentDecision(action, target.id, stance))
    first = comments[0]
    return Decision(first.action, probabilities, first.public_stance, first.target_id, tuple(comments))
