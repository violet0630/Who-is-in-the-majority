"""用途隔离的共同随机数。"""

import hashlib
import json
from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class RandomStreams:
    seed: int
    stream: str = "paired"

    def for_event(self, node: int, round: int, purpose: str) -> Random:
        key = json.dumps([self.seed, self.stream, node, round, purpose]).encode()
        digest = hashlib.sha256(key).digest()
        return Random(int.from_bytes(digest, "big"))
