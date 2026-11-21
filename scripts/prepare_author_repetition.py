"""Freeze paired stimuli with identical messages and different author identity."""

import argparse
import hashlib
import json
from pathlib import Path
from random import Random

from majority.llm import prompt_hash
from majority.llm_perception import perception_payload
from majority.models import AgentState, Profile, Stance
from majority.perception import Exposure, PerceptionRequest, ViewedMessage


def exposure(round_number, author, body):
    message = ViewedMessage(f"message-{round_number}", author, round_number, body, None)
    return Exposure(round_number, (message,), ())


def make_payload(topic, scenario, messages, author_ids, stance, strength):
    exposures = tuple(exposure(i, author, body) for i, (body, author) in
                      enumerate(zip(messages, author_ids, strict=True), 1))
    request = PerceptionRequest(0, Profile(.2, strength), AgentState(stance),
                                exposures[:-1], exposures[-1])
    return perception_payload(topic, request, scenario)


def verify_pair(pair):
    conditions = pair["conditions"]
    repeated, distinct = conditions["repeated_author"], conditions["distinct_authors"]
    for field in ("self", "topic", "scenario"):
        if repeated[field] != distinct[field]:
            raise ValueError(f"{pair['id']}: changed {field} across conditions")
    exposures = [(*payload["history"], payload["current"]) for payload in (repeated, distinct)]
    messages = [[message for item in timeline for message in item["messages"]] for timeline in exposures]
    for first, second in zip(*messages, strict=True):
        if any(first[name] != second[name] for name in ("round", "text", "original_id")):
            raise ValueError(f"{pair['id']}: text or timing changed")
    counter = pair["counter_message_position"]
    focal = [[message["author"] for i, message in enumerate(variant) if i != counter]
             for variant in messages]
    counts = [len(set(authors)) for authors in focal]
    if (counts != [1, 4] or focal[0][0] != focal[1][0]
            or messages[0][counter] != messages[1][counter]
            or any(messages[i][counter]["author"] in focal[i] for i in range(2))
            or any(len({message["id"] for message in variant}) != 5 for variant in messages)):
        raise ValueError(f"{pair['id']}: wrong author identity assignment")


def prepare(config):
    rng = Random(config["seed"])
    pairs = []
    for topic in config["topics"]:
        for focal_stance, focal_key, counter_key in (
            (Stance.SUPPORT, "support", "oppose"), (Stance.OPPOSE, "oppose", "support")
        ):
            focal_bank, counter_bank = topic[focal_key], topic[counter_key]
            if len(focal_bank) != 6 or len(counter_bank) != 6 or len(set(focal_bank + counter_bank)) != 12:
                raise ValueError("Both text banks must contain six distinct messages")
            for i in range(6):
                counter_index = i % 5
                messages = [focal_bank[(i + offset) % 6] for offset in range(4)]
                messages.insert(counter_index, counter_bank[(i + 2) % 6])
                repeated, distinct = [], []
                aliases = rng.sample(range(10, 1000), 5)
                focal_index = 0
                for index in range(5):
                    if index == counter_index:
                        repeated.append(aliases[4])
                        distinct.append(aliases[4])
                    else:
                        repeated.append(aliases[0])
                        distinct.append(aliases[focal_index])
                        focal_index += 1
                stance = focal_stance if i % 2 == 0 else Stance(-int(focal_stance))
                strength = .4 if (len(pairs) // 2) % 2 == 0 else .8
                common = (topic["topic"], config["scenario"], messages)
                pair = {"id": f"{topic['id']}-{focal_key}-{i + 1:02d}", "topic_id": topic["id"],
                        "focal_stance": int(focal_stance), "counter_message_position": counter_index,
                        "conditions": {
                            "repeated_author": make_payload(*common, repeated, stance, strength),
                            "distinct_authors": make_payload(*common, distinct, stance, strength),
                        }}
                verify_pair(pair)
                pairs.append(pair)
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/author_repetition.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.config.read_bytes()
    config = json.loads(source)
    pairs = prepare(config)
    result = {"api_runs": 0, "config_sha256": hashlib.sha256(source).hexdigest(),
              "prompt_sha256": prompt_hash("perception"), "seed": config["seed"],
              "model": config["model"], "temperature": config["temperature"], "pairs": pairs}
    artifact = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_bytes() != artifact:
        raise ValueError("The frozen output differs; use a new versioned path")
    if not args.output.exists():
        with args.output.open("xb") as file:
            file.write(artifact)
    print(f"Verified {len(pairs)} author-identity pairs ({len(pairs) * 2} inputs), 0 API calls")


if __name__ == "__main__":
    main()
