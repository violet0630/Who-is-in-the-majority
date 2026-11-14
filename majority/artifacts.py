"""冻结输入的 JSON 保存/读取；与逐轮 JSONL 分开。"""

import json
from pathlib import Path

from .models import Coefficients, Network, OriginalText, Stance, Text, TextBank
from .storage import write_record


def save_artifact(path: Path, kind: str, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        write_record(file, kind, data)


def load_artifact(path: Path, kind: str) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("kind") != kind or not isinstance(record.get("data"), dict):
        raise ValueError(f"输入文件类型应为 {kind}")
    return record["data"]


def load_network(path: Path) -> Network:
    data = load_artifact(path, "network")["network"]
    return Network(tuple(data["communities"]), tuple(frozenset(group) for group in data["following"]),
                   frozenset(data["observers"]))


def load_bank(path: Path) -> tuple[dict, TextBank]:
    data = load_artifact(path, "text_bank")
    originals = tuple(OriginalText(
        Text(**item["text"]), Stance(item["stance"]),
        tuple(Text(**text) for text in item["agree"]), tuple(Text(**text) for text in item["disagree"]),
    ) for item in data["bank"]["originals"])
    return data["study"], TextBank(originals)


def load_coefficients(path: Path) -> Coefficients:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Coefficients(tuple(data["theta"]), data["kappa"], data["gamma"], data["beta"])
