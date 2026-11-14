"""描述性判断统计；完整运行的因果比较仍由 analysis 的推断接口负责。"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

from .models import Network, Stance
from .simulation import RoundRecord


@dataclass(frozen=True)
class JudgmentSummary:
    branch: str
    round: int
    scope: str
    nodes: int
    updated: int
    carried_forward: int
    mean_support: float
    std_support: float
    support_majority_count: int
    support_majority_fraction: float
    true_support: float
    mean_bias: float
    mean_absolute_error: float
    input_tokens: int
    output_tokens: int
    calls_without_usage: int
    advocate_stance: Stance
    advocate_majority_count: int
    advocate_majority_fraction: float


def summarize_judgments(record: RoundRecord, network: Network, *, observed_only: bool = True) -> JudgmentSummary:
    rows = [row for row in record.nodes if not observed_only or row.node in network.observers]
    values = [row.after.perceived_support for row in rows]
    calls = [row.perception for row in rows if row.perception is not None]
    actual = mean(row.before.stance == Stance.SUPPORT for row in record.nodes)
    majority_count = sum(value > 0.5 for value in values)
    focal_count = sum((value if record.advocate_stance == Stance.SUPPORT else 1 - value) > 0.5
                      for value in values)
    return JudgmentSummary(
        record.branch, record.round, "observers" if observed_only else "all",
        len(rows), len(calls), len(rows) - len(calls), mean(values), pstdev(values),
        majority_count, majority_count / len(rows), actual, mean(values) - actual,
        mean(abs(value - actual) for value in values),
        sum(call.input_tokens for call in calls if call.input_tokens is not None),
        sum(call.output_tokens for call in calls if call.output_tokens is not None),
        sum(call.input_tokens is None or call.output_tokens is None for call in calls),
        record.advocate_stance, focal_count, focal_count / len(rows),
    )


def summarize_calls(path: Path) -> dict[str, dict]:
    groups = defaultdict(list)
    with path.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            if record["kind"] == "llm_call":
                groups[record["data"]["task"]].append(record["data"])
    result = {}
    for task, records in groups.items():
        calls = [record for record in records if not record.get("reused", False)]
        usages = [call.get("response", {}).get("usage") for call in calls]
        result[task] = {
            "reused_records": len(records) - len(calls),
            "calls": len(calls), "successful": sum(call["status"] == "ok" for call in calls),
            "api_errors": sum(call["status"] == "api_error" for call in calls),
            "invalid_responses": sum(call["status"] == "invalid_response" for call in calls),
            "known_input_tokens": sum(usage["prompt_tokens"] for usage in usages
                                      if usage is not None and usage.get("prompt_tokens") is not None),
            "known_output_tokens": sum(usage["completion_tokens"] for usage in usages
                                       if usage is not None and usage.get("completion_tokens") is not None),
            "calls_without_usage": sum(usage is None or usage.get("prompt_tokens") is None
                                       or usage.get("completion_tokens") is None for usage in usages),
            "mean_seconds": mean(call["elapsed_seconds"] for call in calls) if calls else 0,
        }
    return result
