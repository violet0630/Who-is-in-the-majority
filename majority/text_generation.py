"""分批生成并校验冻结文本库；标签由请求条件决定，不由模型编号决定。"""

from .llm import JsonChat
from .models import OriginalText, Stance, Text, TextBank


def _texts(value: object, count: int, min_chars: int, max_chars: int) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError("文本条数不符合要求")
    if any(not isinstance(text, str) or not min_chars <= len(text.strip()) <= max_chars for text in value):
        raise ValueError("文本类型或长度不符合要求")
    texts = tuple(text.strip() for text in value)
    if len(set(texts)) != count:
        raise ValueError("存在重复文本")
    return texts


def generate_text_bank(
    topic: str, chat: JsonChat, *, per_side: int = 50, comments_per_side: int = 5,
    batch_size: int = 10, min_chars: int = 40, max_chars: int = 80, scenario: str = "",
) -> TextBank:
    if not topic.strip() or min(per_side, comments_per_side, batch_size, min_chars) < 1:
        raise ValueError("议题不能为空，条数与长度须为正")
    if max_chars < min_chars:
        raise ValueError("长度上限不能小于下限")
    originals = []
    for stance in Stance:
        generated: list[str] = []
        while len(generated) < per_side:
            count = min(batch_size, per_side - len(generated))

            def parse_originals(data: dict) -> tuple[str, ...]:
                if set(data) != {"texts"}:
                    raise ValueError("原创输出只能包含 texts")
                texts = _texts(data["texts"], count, min_chars, max_chars)
                if set(texts) & set(generated):
                    raise ValueError("原创文本与已生成批次重复")
                return texts

            texts, _ = chat.request("originals", {
                "topic": topic, "scenario": scenario, "stance": int(stance), "count": count,
                "min_chars": min_chars, "max_chars": max_chars, "already_generated": generated.copy(),
            }, parse_originals, temperature=0.7, context={"stance": int(stance), "offset": len(generated)})
            generated.extend(texts)
        for index, body in enumerate(generated):
            id = f"post-{int(stance)}-{index:03d}"

            def parse_comments(data: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
                if set(data) != {"agree", "disagree"}:
                    raise ValueError("评论输出需要 agree 与 disagree")
                agree = _texts(data["agree"], comments_per_side, min_chars, max_chars)
                disagree = _texts(data["disagree"], comments_per_side, min_chars, max_chars)
                if set(agree) & set(disagree):
                    raise ValueError("赞同与反对评论不能相同")
                return agree, disagree

            (agree, disagree), _ = chat.request("comments", {
                "topic": topic, "scenario": scenario, "original": body, "original_stance": int(stance),
                "count_per_side": comments_per_side, "min_chars": min_chars, "max_chars": max_chars,
            }, parse_comments, temperature=0.7, context={"original_id": id})
            originals.append(OriginalText(
                Text(id, body), stance,
                tuple(Text(f"{id}-agree-{i}", text) for i, text in enumerate(agree)),
                tuple(Text(f"{id}-disagree-{i}", text) for i, text in enumerate(disagree)),
            ))
    return TextBank(tuple(originals))
