"""每行一条 JSON 记录；文件句柄由调用者管理。"""

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import TextIO


def _encode(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"不支持序列化 {type(value).__name__}")


def write_record(file: TextIO, kind: str, record: object) -> None:
    file.write(json.dumps({"kind": kind, "data": record}, default=_encode,
                          ensure_ascii=False, allow_nan=False) + "\n")
