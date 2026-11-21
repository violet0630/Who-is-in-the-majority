"""Reproducible, observer-only LLM experiments using the existing majority package."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import hashlib
import gzip
import itertools
import json
from math import exp, log
from pathlib import Path
from random import Random
import sys
import shutil
from threading import BoundedSemaphore
from time import perf_counter, sleep

import numpy as np
from openai import APIConnectionError, APIStatusError
from openai.types.chat import ChatCompletion

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from majority.analysis import measure
from majority.artifacts import load_bank, load_network, save_artifact
from majority.behavior import decide
from majority.llm import Completion, JsonChat, LLMError, LLMValidationError, canonical_model, load_prompt, load_settings, open_client, prompt_hash
from majority.llm_perception import perception_payload
from majority.models import Coefficients, OriginalText, Profile, RunConfig, Stance, Text, TextBank
from majority.network import NetworkConfig, _connected, _observers, network_statistics
from majority.models import Network
from majority.perception import PerceptionResult
from majority.preparation import draw_profiles, seed_diagnostics
from majority.randomness import RandomStreams
from majority.simulation import Simulation
from majority.storage import write_record
from majority.text_generation import _texts

ARCHIVE = PROJECT / "experiments/2026-11-16"
ROOT = ARCHIVE
INITIALIZATION = ROOT / "02_initialization_50"
GATEWAY_SLOTS = BoundedSemaphore(16)


def active_bank():
    local = ROOT / "01_texts_50/bank.json"
    return local if local.exists() else ARCHIVE / "01_texts_50/bank.json"


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def expand_bank(target_per_side=10):
    destination = ROOT / ("01_texts_50" if target_per_side == 25 else "01_texts")
    destination.mkdir(exist_ok=True)
    if (destination / "bank.json").exists():
        raise FileExistsError("The expanded bank is already frozen")
    source = ROOT / "01_texts/bank.json" if target_per_side == 25 else ARCHIVE / "00_texts_original/bank.json"
    if target_per_side == 25 and not source.exists():
        source = ARCHIVE / "01_texts/bank.json"
    study, old = load_bank(source)
    config, key = load_settings(PROJECT / ".env")
    originals = list(old.originals)
    provenance = {
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "llm": asdict(config), "originals_per_side": target_per_side, "comments_per_relation": 2,
        "prompt_hashes": {name: prompt_hash(name) for name in ("originals", "comments", "perception")},
    }
    if not (destination / "source.json").exists():
        save_json(destination / "source.json", provenance)
    audit_path = destination / "calls.jsonl"
    with open_client(config, key) as client, audit_path.open("a", encoding="utf-8") as audit:
        chat = JsonChat(client, config, audit, audit_path)
        for stance in Stance:
            prior = [item.text.body for item in old.originals if item.stance == stance]
            bodies = []
            while len(prior) + len(bodies) < target_per_side:
                count = min(5, target_per_side - len(prior) - len(bodies))
                already = prior + bodies

                def parse_originals(data):
                    if set(data) != {"texts"}:
                        raise ValueError("Expected texts only")
                    texts = _texts(data["texts"], count, 40, 80)
                    if set(texts) & set(already):
                        raise ValueError("New originals duplicate the source bank")
                    return texts

                generated, _ = chat.request("originals", {
                    "topic": study["topic"], "scenario": study["scenario"], "stance": int(stance),
                    "count": count, "min_chars": 40, "max_chars": 80, "already_generated": already,
                }, parse_originals, temperature=0.7,
                    context={"stage": "expansion", "stance": int(stance), "offset": len(already)})
                bodies.extend(generated)

            def with_comments(indexed_body):
                offset, body = indexed_body
                identifier = f"post-{int(stance)}-{offset:03d}"

                def parse_comments(data):
                    if set(data) != {"agree", "disagree"}:
                        raise ValueError("Expected agree/disagree")
                    agree, disagree = (_texts(data[name], 2, 40, 80) for name in ("agree", "disagree"))
                    if set(agree) & set(disagree):
                        raise ValueError("Opposite comments must differ")
                    return agree, disagree

                comment_payload = {
                    "topic": study["topic"], "scenario": study["scenario"], "original": body,
                    "original_stance": int(stance), "count_per_side": 2, "min_chars": 40, "max_chars": 80,
                }
                for attempt in range(3):
                    try:
                        (agree, disagree), _ = chat.request("comments", comment_payload, parse_comments,
                            temperature=0.7, context={"original_id": identifier} if attempt == 0 else
                            {"original_id": identifier, "correction": attempt})
                        break
                    except LLMValidationError:
                        if attempt == 2:
                            raise
                        comment_payload = {**comment_payload, "min_chars": 45, "max_chars": 75,
                            "validation_feedback": "每个数组必须恰好2条字符串；每条按字符串字符数计须45至75字，避免少于40字或超过80字。"}
                original = OriginalText(Text(identifier, body), stance,
                    tuple(Text(f"{identifier}-agree-{i}", value) for i, value in enumerate(agree)),
                    tuple(Text(f"{identifier}-disagree-{i}", value) for i, value in enumerate(disagree)))
                print(f"Expanded {identifier}", flush=True)
                return original

            with ThreadPoolExecutor(max_workers=5) as pool:
                originals.extend(pool.map(with_comments, enumerate(bodies, len(prior))))
    bank = TextBank(tuple(originals))
    save_artifact(destination / "bank.json", "text_bank", {
        "study": study, "bank": bank, "model": config.model,
        "temperature": 0.7, "semantic_review": "pending", "preserved_original_count": len(old.originals),
        "sha256": hashlib.sha256(json.dumps(asdict(bank), ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
    })
    lines = ["# 扩充候选文本", "", f"原创总数{len(originals)}，每条各2条赞同与2条反对评论。", ""]
    for item in bank.originals:
        lines.extend([f"## {item.text.id} · 议题立场 {int(item.stance):+d}", "", item.text.body, ""])
        for relation in ("agree", "disagree"):
            lines.extend(f"- {text.id}: {text.body}" for text in getattr(item, relation))
        lines.append("")
    (destination / "candidate_pool.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Text bank complete: {len(originals)} originals, {len(originals) * 4} comments", flush=True)


class ObserverPerception:
    def __init__(self, observers, bank, llm=None):
        self.observers, self.llm = observers, llm
        self.labels = {}
        for original in bank.originals:
            for text in (original.text, *original.agree):
                self.labels[text.body] = int(original.stance)
            for text in original.disagree:
                self.labels[text.body] = -int(original.stance)

    def estimate(self, request):
        if request.node not in self.observers:
            authors = {}
            for message in request.current.messages:
                authors[message.author] = self.labels[message.text]
            support = (sum(stance == 1 for stance in authors.values()) / len(authors)
                       if authors else request.state.perceived_support)
            return PerceptionResult(support, json.dumps({"support": support, "source": "local_proxy"}),
                                    model="local_proxy")
        if self.llm:
            return self.llm.estimate(request)
        # Low-fidelity screening uses only directly read messages, with authors deduplicated.
        authors = {}
        for exposure in (*request.history, request.current):
            for message in exposure.messages:
                authors[message.author] = self.labels[message.text]
        support = (1 + sum(stance == 1 for stance in authors.values())) / (2 + len(authors))
        return PerceptionResult(support, json.dumps({"support": support}), model="screening_author_counts")


class StudyChat(JsonChat):
    """Use documented non-thinking mode and include it in audit/cache identities."""
    def __init__(self, client, config, audit, reuse_log=None):
        super().__init__(client, config, audit, reuse_log)
        self.invalid_reuse = {}
        self.cache_source_models = {}
        if reuse_log:
            legacy_ok, legacy_invalid = {}, {}
            with reuse_log.open(encoding="utf-8") as stream:
                for line in stream:
                    item = json.loads(line)["data"]
                    if item["status"] == "invalid_response":
                        key = json.dumps([item["task"], item["payload"], item["context"], item["prompt_sha256"],
                                          canonical_model(item["model"]), item["temperature"], item["max_tokens"]], sort_keys=True)
                        self.invalid_reuse[key] = item["response"]
                    if config.model == "deepseek-v4-flash" and item["model"] == "deepseek-flash":
                        if item["status"] not in ("ok", "invalid_response"):
                            continue
                        key = json.dumps([item["task"], item["payload"], item["context"], item["prompt_sha256"],
                                          canonical_model(config.model), item["temperature"], item["max_tokens"]], sort_keys=True)
                        target = legacy_ok if item["status"] == "ok" else legacy_invalid
                        target[key] = item["response"]
            for key, response in legacy_invalid.items():
                if key not in legacy_ok:
                    self.reuse.pop(key, None)
                    self.invalid_reuse[key] = response
                    self.cache_source_models[key] = "deepseek-flash"
            for key, response in legacy_ok.items():
                self.reuse[key] = response
                self.invalid_reuse.pop(key, None)
                self.cache_source_models[key] = "deepseek-flash"

    def request(self, prompt, payload, parse, *, temperature, context=None):
        for attempt in range(3):
            try:
                return self._request_once(prompt, payload, parse, temperature=temperature, context=context)
            except LLMValidationError:
                raise
            except LLMError:
                if attempt == 2:
                    raise
                sleep(1 + 2 * attempt)

    def _request_once(self, prompt, payload, parse, *, temperature, context=None):
        system = load_prompt(prompt)
        context = {**(context or {}), "thinking": "disabled"}
        record = {"task": prompt, "context": context, "model": self.config.model,
                  "canonical_model": canonical_model(self.config.model),
                  "prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
                  "temperature": temperature, "max_tokens": self.config.max_tokens,
                  "extra_body": {"thinking": {"type": "disabled"}}, "payload": payload}
        cache_key = json.dumps([prompt, payload, context, record["prompt_sha256"],
                                canonical_model(self.config.model), temperature, self.config.max_tokens], sort_keys=True)
        start = perf_counter()
        record["reused"] = cache_key in self.reuse or cache_key in self.invalid_reuse
        if record["reused"]:
            record["cache_source_model"] = self.cache_source_models.get(cache_key, self.config.model)
        try:
            if record["reused"]:
                saved = self.reuse.get(cache_key) or self.invalid_reuse[cache_key]
                response = ChatCompletion.model_validate(saved)
            else:
                with GATEWAY_SLOTS:
                    response = self.client.chat.completions.create(model=self.config.model,
                        messages=[{"role": "system", "content": system},
                                  {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                        response_format={"type": "json_object"}, temperature=temperature,
                        max_tokens=self.config.max_tokens, extra_body=record["extra_body"])
        except (APIConnectionError, APIStatusError) as exc:
            record.update(status="api_error", error=type(exc).__name__, elapsed_seconds=perf_counter() - start)
            self._record(record)
            raise LLMError(f"{prompt}: {type(exc).__name__}; see audit log") from None
        elapsed = perf_counter() - start
        record.update(response=response.model_dump(mode="json"), elapsed_seconds=elapsed)
        try:
            if not response.choices or response.choices[0].finish_reason != "stop":
                raise ValueError("Incomplete response")
            content = response.choices[0].message.content
            data = json.loads(content)
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            parsed = parse(data)
        except (ValueError, KeyError, TypeError) as exc:
            record.update(status="invalid_response", error=str(exc))
            self._record(record)
            raise LLMValidationError(str(exc)) from None
        record["status"] = "ok"
        self._record(record)
        usage = response.usage
        return parsed, Completion(content, response.id, response.model,
            usage.prompt_tokens if usage else None, usage.completion_tokens if usage else None, elapsed)


class StudyPerception:
    def __init__(self, chat, topic, scenario, run_id):
        self.chat, self.topic, self.scenario, self.run_id = chat, topic, scenario, run_id

    def estimate(self, request):
        payload = perception_payload(self.topic, request, self.scenario)
        allowed = {message["id"] for exposure in (*payload["history"], payload["current"])
                   for message in (*exposure["messages"], *exposure["context"])}
        payload["allowed_evidence_ids"] = sorted(allowed)
        payload["output_guidance"] = "evidence_ids只能从allowed_evidence_ids选取，也可返回空数组；不能自行拼写不存在的编号。"

        def parse(data):
            if set(data) != {"support", "reason", "evidence_ids"}:
                raise ValueError("Exactly support, reason, evidence_ids are required")
            support, reason, evidence = data["support"], data["reason"], data["evidence_ids"]
            if type(support) not in (int, float) or not 0 <= support <= 1:
                raise ValueError("support must be a number between 0 and 1")
            if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 120:
                raise ValueError("reason must contain 1 to 120 characters")
            if not isinstance(evidence, list) or len(evidence) > 5 or any(not isinstance(x, str) for x in evidence):
                raise ValueError("evidence_ids must contain at most five strings")
            if len(set(evidence)) != len(evidence) or not set(evidence) <= allowed:
                raise ValueError(f"Unknown or duplicate evidence IDs: {sorted(set(evidence) - allowed)}")
            return float(support), reason, tuple(evidence)

        for attempt in range(3):
            try:
                parsed, completion = self.chat.request("perception", payload, parse, temperature=0,
                    context={"run_id": self.run_id, "node": request.node, "round": request.current.round,
                             "schema_attempt": attempt})
                break
            except LLMValidationError as exc:
                if attempt == 2:
                    raise
                payload = {**payload, "validation_feedback": str(exc) + "; evidence_ids可设为空数组。重新输出三个字段的JSON。"}
        support, reason, evidence = parsed
        return PerceptionResult(support, completion.raw_json, completion.input_tokens, completion.output_tokens,
                                reason, evidence, completion.request_id, completion.model,
                                prompt_hash("perception"), completion.elapsed_seconds)


class ObserverSimulation(Simulation):
    pass


def network_config(size, settings):
    edges = round(size * settings["mean_degree"])
    fractions = settings.get("community_fractions", (0.15, 0.20, 0.25, 0.40))
    return NetworkConfig(community_sizes=tuple(round(size * x) for x in fractions),
                         s_out=settings.get("s_out", 0.35), s_in=settings["s_in"],
                         between_weight=settings["w"], expected_edges=edges,
                         edge_range=(round(edges * 0.95), round(edges * 1.05)), observer_count=50)


def coefficients_from(settings):
    return Coefficients(tuple(settings["theta"]), settings["kappa"], settings["gamma"], beta=10.0)


def make_network(size, settings, seed):
    """Sample the existing edge model, conditioning each row on positive outdegree."""
    config = network_config(size, settings)
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(4), config.community_sizes)
    rng.shuffle(labels)
    out = rng.lognormal(-config.s_out ** 2 / 2, config.s_out, size)
    incoming = rng.lognormal(-config.s_in ** 2 / 2, config.s_in, size)
    weights = np.outer(out, incoming) * np.where(labels[:, None] == labels[None, :], 1.0, config.between_weight)
    np.fill_diagonal(weights, 0)
    lower, upper = 0.0, 1.0
    while (-np.expm1(-upper * weights)).sum() < config.expected_edges:
        upper *= 2
    for _ in range(45):
        middle = (lower + upper) / 2
        if (-np.expm1(-middle * weights)).sum() < config.expected_edges:
            lower = middle
        else:
            upper = middle
    probabilities = -np.expm1(-(lower + upper) / 2 * weights)
    communities = tuple(map(int, labels))
    for _ in range(config.max_attempts):
        adjacency = rng.random((size, size)) < probabilities
        empty = np.flatnonzero(~adjacency.any(axis=1))
        while len(empty):
            adjacency[empty] = rng.random((len(empty), size)) < probabilities[empty]
            empty = np.flatnonzero(~adjacency.any(axis=1))
        edge_count = int(adjacency.sum())
        if not config.edge_range[0] <= edge_count <= config.edge_range[1]:
            continue
        following = tuple(frozenset(map(int, np.flatnonzero(row))) for row in adjacency)
        if _connected(following):
            observers = _observers(communities, following, config.observer_count, Random(seed + 17))
            return Network(communities, following, observers)
    raise RuntimeError("Network constraints not reached")


def select_positions(network, profiles, k, seed, steps=5000, restarts=4, rho=1.0):
    """Matched annealing search; swaps preserve community and strength composition."""
    rng = Random(seed)
    size = len(profiles)
    adjacency = np.zeros((size, size), dtype=np.int16)
    for node, following in enumerate(network.following):
        adjacency[node, list(following)] = 1
    degrees = adjacency.sum(axis=1)
    fans = adjacency.sum(axis=0)
    baseline = np.array([profile.baseline for profile in profiles])
    boosted = rho * baseline / (1 - baseline + rho * baseline)
    groups = [sorted(i for i, label in enumerate(network.communities)
                     if label == community and i not in network.observers) for community in range(4)]
    available = sum(map(len, groups))
    quotas = [k * len(group) // available for group in groups]
    order = sorted(range(4), key=lambda i: (-(k * len(groups[i]) % available), i))
    for index in order[:k - sum(quotas)]:
        quotas[index] += 1
    reference = frozenset(node for group, quota in zip(groups, quotas) for node in rng.sample(group, quota))
    cells = [tuple(i for i in group if profiles[i].strength == strength)
             for group in groups for strength in (0.4, 0.8)]
    cells = [cell for cell in cells if 0 < len(reference.intersection(cell)) < len(cell)]
    reference_fans = float(fans[list(reference)].sum())
    reference_baseline = float(baseline[list(reference)].sum())
    reference_boosted = float(boosted[list(reference)].sum())

    def score(counts, mask):
        ordinary = ~mask
        illusion = float(np.mean(counts[ordinary] > degrees[ordinary] / 2))
        smooth = float(np.mean((counts[ordinary] / degrees[ordinary]) ** 3))
        return illusion + 0.01 * smooth

    selections = {"U0": reference}
    for branch, direction in (("TH", 1), ("TL", -1)):
        best = reference
        mask = np.zeros(size, dtype=bool)
        mask[list(reference)] = True
        best_score = score(adjacency[:, list(reference)].sum(axis=1), mask)
        for _ in range(restarts):
            current = set(reference)
            mask = np.zeros(size, dtype=bool)
            mask[list(current)] = True
            counts = adjacency[:, list(current)].sum(axis=1)
            fan_total, base_total, boosted_total = reference_fans, reference_baseline, reference_boosted
            value = score(counts, mask)
            for step in range(steps):
                cell = cells[rng.randrange(len(cells))]
                outgoing = rng.choice([i for i in cell if i in current])
                incoming = rng.choice([i for i in cell if i not in current])
                new_fans = fan_total - int(fans[outgoing]) + int(fans[incoming])
                new_base = base_total - baseline[outgoing] + baseline[incoming]
                new_boosted = boosted_total - boosted[outgoing] + boosted[incoming]
                if any(abs(new - old) > 0.024 * old for new, old in
                       ((new_fans, reference_fans), (new_base, reference_baseline),
                        (new_boosted, reference_boosted))):
                    continue
                new_counts = counts - adjacency[:, outgoing] + adjacency[:, incoming]
                mask[outgoing], mask[incoming] = False, True
                new_value = score(new_counts, mask)
                temperature = max(0.0001, 0.004 * (1 - step / steps) ** 2)
                accept = direction * (new_value - value) >= 0 or rng.random() < exp(direction * (new_value - value) / temperature)
                if accept:
                    current.remove(outgoing)
                    current.add(incoming)
                    counts, fan_total, base_total, boosted_total, value = new_counts, new_fans, new_base, new_boosted, new_value
                    if direction * (value - best_score) > 0:
                        best, best_score = frozenset(current), value
                else:
                    mask[outgoing], mask[incoming] = True, False
        selections[branch] = best
    diagnostics = {branch: asdict(seed_diagnostics(network, profiles, selected)) for branch, selected in selections.items()}
    for branch, selected in selections.items():
        diagnostics[branch]["observer_illusion"] = sum(len(network.following[i] & selected) > len(network.following[i]) / 2
                                                         for i in network.observers) / len(network.observers)
        diagnostics[branch]["advocate_strong_count"] = sum(profiles[i].strength == 0.8 for i in selected)
        diagnostics[branch]["initial_expected_advocate_speakers"] = float(boosted[list(selected)].sum())
    return selections, diagnostics


def screening():
    destination = INITIALIZATION
    destination.mkdir(exist_ok=False)
    _, bank = load_bank(active_bank())
    rng = Random(2026111601)
    candidates = [{"mean_degree": 8, "s_out": 0.35, "s_in": 1.0, "w": 0.2,
                   "fraction": 0.16, "rho": 3, "kappa": 1, "gamma": 1, "theta": [0, 0, 1, 0], "rounds": 6}]
    for _ in range(95):
        candidates.append({"mean_degree": rng.choice([4, 6, 8, 12]),
            "s_out": rng.choice([0.2, 0.35, 0.5]), "s_in": rng.choice([0.5, 1.0, 1.5]),
            "w": rng.choice([0.05, 0.2, 0.5]), "fraction": rng.choice([0.08, 0.12, 0.16, 0.20]),
            "rho": rng.choice([1, 3, 6]), "kappa": rng.choice([0.5, 1, 1.5, 2]),
            "gamma": rng.choice([0.5, 1, 2]),
            "theta": [rng.choice([-1, 0, 1]), rng.choice([-0.5, 0, 0.5]), rng.choice([-1, 0, 1]), rng.choice([-1, 0, 1])],
            "rounds": 6})
    save_json(destination / "search_design.json", {
        "method": "random search with successive halving; observer author-count proxy, real LLM confirmation follows",
        "seed": 2026111601, "size": 100, "observers": 50, "alpha": 0.05, "beta": 10,
        "candidate_count": len(candidates), "candidates": candidates,
        "selection": "Prefer tail Q >= .25; penalize larger minority and speech advantage; do not optimize TH-TL differences",
    })
    results = []
    with (destination / "screening.jsonl").open("x", encoding="utf-8") as stream:
        for index, settings in enumerate(candidates):
            network = make_network(100, settings, 4100 + index)
            profiles = draw_profiles(100, Random(5200 + index))
            positions, diagnostics = select_positions(network, profiles, round(100 * settings["fraction"]),
                                                       6300 + index, steps=600, restarts=2)
            runs = [run_one(network, profiles, positions[branch], bank, settings, 7400 + index, branch)
                    for branch in ("TH", "U0")]
            value = sum(run["summary"]["perceived_majority"] for run in runs) / len(runs)
            result = {"candidate": index, "settings": settings, "proxy_q": value,
                      "score": value - 0.4 * settings["fraction"] - 0.015 * log(settings["rho"]),
                      "position_diagnostics": diagnostics, "runs": runs}
            results.append(result)
            write_record(stream, "screening", result)
            stream.flush()
            print(f"screen {index+1}/{len(candidates)} Q={value:.3f} score={result['score']:.3f}", flush=True)
    ranked = sorted(results, key=lambda row: row["score"], reverse=True)
    finalists = []
    for rank, item in enumerate(ranked[:8]):
        settings = {**item["settings"], "rounds": 12}
        runs = []
        for repeat in range(3):
            network = make_network(100, settings, 8100 + repeat)
            profiles = draw_profiles(100, Random(8200 + repeat))
            positions, _ = select_positions(network, profiles, round(100 * settings["fraction"]),
                                             8300 + repeat, steps=1500, restarts=2)
            for branch in ("TH", "U0"):
                runs.append(run_one(network, profiles, positions[branch], bank, settings, 8400 + repeat, branch))
        value = sum(run["summary"]["perceived_majority"] for run in runs) / len(runs)
        finalist = {"candidate": item["candidate"], "settings": settings, "proxy_q": value,
                    "score": value - 0.4 * settings["fraction"] - 0.015 * log(settings["rho"]), "runs": runs}
        finalists.append(finalist)
        print(f"finalist {rank+1}/8 Q={value:.3f}", flush=True)
    save_json(destination / "finalists.json", sorted(finalists, key=lambda row: row["score"], reverse=True))


def run_one(network, profiles, advocates, bank, settings, seed, branch, output=None, llm=False):
    study = json.loads((PROJECT / "configs/study.json").read_text(encoding="utf-8"))
    config = RunConfig(rounds=settings["rounds"], alpha=0.05, rho=settings["rho"],
                       advocate_stance=Stance.OPPOSE, window=settings.get("window", 2),
                       read_limit=settings.get("read_limit", 11),
                       cross_fan_comments=settings.get("cross_fan_comments", True))
    coefficients = coefficients_from(settings)
    metrics, diagnostic_rows = [], []
    prefix_reference = None
    prefix_path = ROOT / "03_positions/prefix8/results.json"
    if llm and output.parent == ROOT / "03_positions" and prefix_path.exists():
        prefix_rows = json.loads(prefix_path.read_text(encoding="utf-8"))
        prefix_reference = next(row["metrics"] for row in prefix_rows
                                if output.name == f"g{row['graph']}-r{row['repeat']}-{row['branch']}")
    client = audit = trajectory = None
    try:
        perception_llm = None
        if llm:
            if (output / "result.json").exists():
                return json.loads((output / "result.json").read_text(encoding="utf-8"))
            output.mkdir(parents=True, exist_ok=True)
            llm_config, key = load_settings(PROJECT / ".env")
            llm_config = replace(llm_config, timeout=180)
            client = open_client(llm_config, key)
            audit_path = output / "calls.jsonl"
            audit = audit_path.open("a", encoding="utf-8")
            chat = StudyChat(client, llm_config, audit, audit_path)
            perception_llm = StudyPerception(chat, study["topic"], study["scenario"], output.name)
            trajectory = gzip.open(output / "trajectory.partial.jsonl.gz", "wt", encoding="utf-8")
            write_record(trajectory, "manifest", {
                "network": network, "profiles": profiles, "advocates": advocates,
                "config": config, "coefficients": coefficients, "group_seed": seed, "branch": branch,
                "observer_count": len(network.observers), "llm": llm_config,
                "perception_scope": "observers use LLM; nonobservers use local_proxy",
                "settings": settings, "prompt_sha256": prompt_hash("perception"), "thinking": "disabled",
            })
        perception = ObserverPerception(network.observers, bank, perception_llm)
        simulation = ObserverSimulation(network, profiles, advocates, bank, coefficients, config,
                                        perception, RandomStreams(seed, "paired"), branch,
                                        workers=settings.get("llm_workers", 50) if llm else 1)
        for record in simulation.run():
            metric = asdict(measure(record, network, simulation.messages))
            if prefix_reference is not None and record.round <= 8:
                assert metric == prefix_reference[record.round - 1], f"Recovered prefix differs: {output.name}, round {record.round}"
            metrics.append(metric)
            observed = [row for row in record.nodes if row.node in network.observers]
            diagnostic = {"round": record.round,
                "active_observers": sum(bool(row.reading.read) for row in observed),
                "mean_perceived_advocate": sum(1 - row.after.perceived_support for row in observed) / len(observed),
                "mean_feedback_observers": sum(row.after.feedback_value for row in observed) / len(observed),
                "messages": len(record.published)}
            diagnostic_rows.append(diagnostic)
            if trajectory:
                write_record(trajectory, "round", record)
                write_record(trajectory, "metrics", metric)
                write_record(trajectory, "diagnostics", diagnostic)
                trajectory.flush()
                print(f"{output.name} round={record.round}/{config.rounds} Q={metric['perceived_majority']:.3f} J={metric['feed_majority']:.3f}", flush=True)
        last = min(settings.get("tail_rounds", 3), config.rounds)
        summary = {name: sum(row[name] for row in metrics[-last:]) / last for name in (
            "perceived_majority", "feed_majority", "advocate_perception_bias", "public_conformity", "opponent_silence")}
        result = {"branch": branch, "seed": seed, "size": len(profiles), "k": len(advocates),
                  "settings": settings, "tail_rounds": last, "summary": summary,
                  "metrics": metrics, "diagnostics": diagnostic_rows,
                  "seed_diagnostics": asdict(seed_diagnostics(network, profiles, advocates))}
        if output:
            save_json(output / "result.json", result)
            trajectory.close()
            trajectory = None
            (output / "trajectory.partial.jsonl.gz").rename(output / "trajectory.jsonl.gz")
        return result
    finally:
        for resource in (trajectory, audit, client):
            if resource:
                resource.close()


def confirm_initialization():
    destination = INITIALIZATION
    finalists = json.loads((destination / "finalists.json").read_text(encoding="utf-8"))
    design = json.loads((destination / "search_design.json").read_text(encoding="utf-8"))
    _, bank = load_bank(active_bank())
    candidates = [("baseline", design["candidates"][0]),
                  *[(f"candidate{item['candidate']}", item["settings"]) for item in finalists[:2]]]
    def execute_candidate(named_candidate):
        name, candidate = named_candidate
        settings = {**candidate, "rounds": 6, "llm_workers": 16}
        network = make_network(100, settings, 9101)
        profiles = draw_profiles(100, Random(9201))
        positions, diagnostics = select_positions(network, profiles, round(100 * settings["fraction"]), 9301,
                                                  steps=3000, restarts=3)
        output = destination / "llm_confirmation_fast" / name
        output.parent.mkdir(parents=True, exist_ok=True)
        if not (output.parent / f"{name}_positions.json").exists():
            save_json(output.parent / f"{name}_positions.json", {"positions": {b: sorted(v) for b, v in positions.items()},
                                                                 "diagnostics": diagnostics})
        result = run_one(network, profiles, positions["TH"], bank, settings, 9401, "TH", output, True)
        return {"candidate": name, **result}
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(execute_candidate, candidates))
    save_json(destination / "llm_confirmation.json", results)


def explore_controls():
    finalists = json.loads((INITIALIZATION / "finalists.json").read_text(encoding="utf-8"))
    baseline = finalists[0]["settings"]
    _, bank = load_bank(active_bank())
    grids = {"window": [1, 2, 3], "read_limit": [3, 5, 8], "rounds": [6, 12, 18],
             "community_fractions": [[0.15, 0.2, 0.25, 0.4], [0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]],
             "strong_fraction": [0.25, 0.5, 0.75], "cross_fan_comments": [False, True]}
    results = []
    for parameter, values in grids.items():
        for value in values:
            settings = {**baseline, parameter: value}
            runs = []
            for repeat in range(3):
                network = make_network(100, settings, 21000 + repeat)
                profiles = draw_profiles(100, Random(21100 + repeat))
                if parameter == "strong_fraction" and value != 0.5:
                    order = list(range(100))
                    Random(21200 + repeat).shuffle(order)
                    strong = set(order[:round(100 * value)])
                    profiles = tuple(replace(profile, strength=0.8 if i in strong else 0.4)
                                     for i, profile in enumerate(profiles))
                positions, _ = select_positions(network, profiles, round(100 * settings["fraction"]),
                    21300 + repeat, steps=1000, restarts=2, rho=settings["rho"])
                runs.append(run_one(network, profiles, positions["TH"], bank, settings, 21400 + repeat, "TH"))
            result = {"parameter": parameter, "value": value, "runs": runs,
                      "proxy_q": sum(run["summary"]["perceived_majority"] for run in runs) / 3}
            results.append(result)
            print(f"control {parameter}={value}: Q={result['proxy_q']:.3f}", flush=True)
    save_json(INITIALIZATION / "controls.json", {"grids": grids, "results": results})


def refine_initialization():
    finalist = json.loads((INITIALIZATION / "finalists.json").read_text(encoding="utf-8"))[0]
    _, bank = load_bank(active_bank())
    candidates = [{**finalist["settings"], "fraction": fraction, "rounds": 6, "llm_workers": 24}
                  for fraction in (0.20, 0.24)]
    if not (INITIALIZATION / "refinement_design.json").exists():
        save_json(INITIALIZATION / "refinement_design.json", {
            "reason": "increase minority fraction moderately while keeping the simpler stable candidate's remaining parameters",
            "target": "tail Q >= .25; prefer the smaller fraction that reaches this observability target",
            "candidates": candidates})

    def execute(settings):
        name = f"fraction{round(100 * settings['fraction'])}"
        network = make_network(100, settings, 9101)
        profiles = draw_profiles(100, Random(9201))
        positions, diagnostics = select_positions(network, profiles, round(100 * settings["fraction"]),
                                                  9301, steps=3000, restarts=3, rho=settings["rho"])
        result = run_one(network, profiles, positions["TH"], bank, settings, 9401, "TH",
                         INITIALIZATION / "llm_refinement" / name, True)
        return {"candidate": name, "position_diagnostics": diagnostics, **result}

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(execute, candidates))
    save_json(INITIALIZATION / "llm_refinement.json", results)


def freeze_initialization():
    candidates = json.loads((INITIALIZATION / "llm_confirmation.json").read_text(encoding="utf-8"))
    candidates += json.loads((INITIALIZATION / "llm_refinement.json").read_text(encoding="utf-8"))
    eligible = [row for row in candidates if row["summary"]["perceived_majority"] >= 0.25]
    selected = min(eligible, key=lambda row: (row["settings"]["fraction"], row["settings"]["rho"])) if eligible else max(
        candidates, key=lambda row: row["summary"]["perceived_majority"])
    settings = {**selected["settings"], "rounds": 10, "tail_rounds": 3, "llm_workers": 4,
                "window": 2, "read_limit": 5, "cross_fan_comments": True,
                "community_fractions": [0.15, 0.20, 0.25, 0.40]}
    initialized = {"size": 1000, "observer_count": 50, "settings": settings,
        "k": round(1000 * settings["fraction"]), "expected_edges": round(1000 * settings["mean_degree"]),
        "alpha": 0.05, "beta": 10, "baseline_distribution": {"family": "Beta", "a": 0.24, "b": 0.85},
        "strength_values": [0.4, 0.8], "strong_fraction": 0.5,
        "selected_candidate": selected["candidate"], "initialization_Q": selected["summary"]["perceived_majority"],
        "selection_rule": "smallest minority among real-LLM candidates with tail Q >= .25",
        "text_bank": str(active_bank()), "text_bank_sha256": json.loads(active_bank().read_text(encoding="utf-8"))["data"]["sha256"],
        "observer_rule": "50 degree/community-stratified nodes; exclude them from advocates",
        "nonobserver_rule": "no perception estimation; gamma=theta_d=theta_hd=0; private stance and feedback drive actions",
        "llm": {"canonical_model": "deepseek-flash", "gateway_models": {
                    "https://api.1nvictus.me/v1": "deepseek-flash",
                    "http://127.0.0.1:3000/v1": "deepseek-v4-flash"}, "temperature": 0, "thinking": "disabled",
                "timeout_seconds": 180, "transport_attempts": 3, "schema_attempts": 3,
                "workers_per_run": 4, "concurrent_runs": 27, "history": "complete personal reading history"}}
    save_json(ROOT / "initialized_parameters.json", initialized)
    save_json(ROOT / "coefficients.initialized.json", {
        "theta": settings["theta"], "kappa": settings["kappa"], "gamma": settings["gamma"], "beta": 10})
    parameters = [
        ("N", 1000, "用户指定", "1000"), ("observer_count", 50, "用户指定", "50"),
        ("k", initialized["k"], "真实LLM初始化搜索", "N×[0.08,0.12,0.16,0.20,0.24]"),
        ("expected_edges", initialized["expected_edges"], "初始化搜索；E=N×平均关注数", "N×[4,6,8,12]"),
        ("rounds", 10, "用户5—10轮；同一轨迹比较5/8/10轮", "5—10"),
        ("tail_rounds", 3, "预先固定第8—10轮", "3"),
        ("alpha", 0.05, "Banisch & Olbrich 2019，默认仿真学习率", "文献初始化0.05"),
        ("beta", 10, "Gaisbauer et al. 2020，数值分析设定", "文献初始化10"),
        ("g0", "Beta(0.24,0.85)", "Román et al. 2012；停止概率Beta(0.85,0.24)的补概率", "文献分布转换"),
        ("h", [0.4, 0.8], "原方案的两档强度量表", "通过κ搜索整体作用强度"),
        ("strong_fraction", 0.5, "两档各半；邻域搜索", "[0.25,0.5,0.75]"),
        ("rho", settings["rho"], "初始化搜索；发言优势比", "[1,3,6]"),
        ("kappa", settings["kappa"], "初始化搜索", "[0.5,1,1.5,2]"),
        ("gamma_observers", settings["gamma"], "初始化搜索", "[0.5,1,2]"),
        ("theta", settings["theta"], "初始化搜索；顺序为截距、强度、逆风、交互", "θ0/θd/θhd∈[-1,0,1]，θh∈[-0.5,0,0.5]"),
        ("s_out", settings["s_out"], "初始化搜索", "[0.2,0.35,0.5]"),
        ("s_in", settings["s_in"], "初始化搜索", "[0.5,1,1.5]"),
        ("between_weight", settings["w"], "初始化搜索", "[0.05,0.2,0.5]"),
        ("community_sizes", [150, 200, 250, 400], "原方案同比放大；邻域搜索", "[150,200,250,400] / [250,250,250,250] / [100,200,300,400]"),
        ("lambda", "逐图二分求解", "使边概率之和等于expected_edges", "派生量"),
        ("edge_range", [round(initialized["expected_edges"]*0.95), round(initialized["expected_edges"]*1.05)], "固定5%生成容差", "±5%"),
        ("read_window", 2, "邻域搜索；保留原方案起点", "[1,2,3]"),
        ("read_limit", 5, "邻域搜索；保留原方案起点", "[3,5,8]"),
        ("cross_fan_comments", True, "保留评论跨粉丝传播；邻域搜索", "[false,true]"),
        ("recommendation_eta", 0, "关系传播模型", "固定0"),
        ("private_attitude_update", False, "固定私人立场主实验", "固定false"),
        ("initial_Q", 0, "无反馈初始值", "固定0"),
        ("initial_perceived_support", 0.5, "中性初始判断；严格大于一半才算多数", "固定0.5"),
        ("originals_per_side", 25, "用户指定", "25"),
        ("comments_per_relation", 2, "用户指定", "2"),
        ("text_chars", [40, 80], "原方案文本协议", "40—80"),
        ("generation_temperature", 0.7, "固定生成协议", "0.7"),
        ("perception_temperature", 0, "固定感知协议", "0"),
        ("seed_match_tolerance", 0.05, "传播资源配对协议", "5%"),
        ("seed_search_steps_restarts", [20000, 4], "结构搜索强度复核", "6000×4 / 20000×4"),
        ("bootstrap_repeats", 10000, "预先固定统计协议", "10000"),
        ("confidence_level", 0.95, "预先固定统计协议", "95%"),
        ("test_alpha", 0.05, "Holm校正后的显著性标准", "0.05"),
        ("coverage_target_q", 0.8, "保留方案阈值定义；本次分析误判比例及位置效应", "0.8"),
    ]
    with (ROOT / "parameters.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["parameter", "initialized_value", "basis", "explored_or_fixed_range"])
        writer.writerows((name, json.dumps(value, ensure_ascii=False), basis, domain)
                         for name, value, basis, domain in parameters)
    print(json.dumps(initialized, ensure_ascii=False, indent=2), flush=True)


def position_experiment():
    destination = ROOT / "03_positions"
    if (destination / "results.json").exists():
        print("Completed results already exist; use --output for a new run.")
        return
    initialization_path = ROOT / "initialized_parameters.json"
    if not initialization_path.exists():
        shutil.copy2(PROJECT / "configs/initialized_20261116.json", initialization_path)
    initialization = json.loads(initialization_path.read_text(encoding="utf-8"))
    settings = initialization["settings"]
    destination.mkdir(parents=True, exist_ok=True)
    protocol = {"size": 1000, "observers": 50, "networks": 3, "groups_per_network": 3,
        "branches": ["TH", "TL", "U0"], "settings": settings,
        "primary_outcome": "mean observer perceived_majority over rounds 8 to 10",
        "comparisons": [["TH", "TL"], ["TH", "U0"], ["TL", "U0"]],
        "statistics": "10000 graph-stratified paired-group bootstrap draws; exact paired sign-flip test; Holm adjustment",
        "time_analysis": "compare mean Q in rounds 8-10 with rounds 3-5; Holm adjustment across three branches",
        "inference_population": "independent profile/process groups conditional on the three generated networks",
        "selection_rule": "choose branch with largest graph-equally-weighted mean Q as subsequent position initialization",
        "seed_pairing": "5% maximum pairwise fan-total, baseline-total, and rho-adjusted initial speech-total difference; preserve community/strength quotas",
        "seed_objective": "structural IG on all ordinary nodes, not observer LLM outcomes"}
    if not (destination / "protocol.json").exists():
        save_json(destination / "protocol.json", protocol)
    else:
        assert json.loads((destination / "protocol.json").read_text(encoding="utf-8")) == protocol
    _, bank = load_bank(active_bank())
    jobs = []
    for graph in range(3):
        network_path = destination / f"network-{graph}.json"
        if network_path.exists():
            network = load_network(network_path)
        else:
            network = make_network(1000, settings, 15000 + graph)
            save_artifact(network_path, "network", {"seed": 15000 + graph,
                "config": network_config(1000, settings), "network": network,
                "statistics": network_statistics(network), "sampler": "rowwise rejection conditional on positive outdegree"})
        for repeat in range(3):
            group_id = f"g{graph}-r{repeat}"
            input_path = destination / f"{group_id}_positions.json"
            if input_path.exists():
                prepared = json.loads(input_path.read_text(encoding="utf-8"))
                profiles = tuple(Profile(**row) for row in prepared["profiles"])
                positions = {name: frozenset(nodes) for name, nodes in prepared["positions"].items()}
                diagnostics = prepared["diagnostics"]
            else:
                profiles = draw_profiles(1000, Random(16000 + graph * 10 + repeat))
                positions, diagnostics = select_positions(network, profiles, round(1000 * settings["fraction"]),
                    17000 + graph * 10 + repeat, steps=20000, restarts=4, rho=settings["rho"])
                save_json(input_path, {"graph": graph, "repeat": repeat, "profiles": [asdict(p) for p in profiles],
                    "positions": {name: sorted(nodes) for name, nodes in positions.items()}, "diagnostics": diagnostics})
            print(f"{group_id} IG high={diagnostics['TH']['illusion']:.3f} low={diagnostics['TL']['illusion']:.3f}", flush=True)
            order = ["TH", "TL", "U0"]
            order = order[repeat:] + order[:repeat]
            jobs.extend((graph, repeat, group_id, network, profiles, positions, branch) for branch in order)

    def execute_run(job):
        graph, repeat, group_id, network, profiles, positions, branch = job
        run = run_one(network, profiles, positions[branch], bank, settings,
            18000 + graph * 10 + repeat, branch, destination / f"{group_id}-{branch}", True)
        return {"graph": graph, "repeat": repeat, **run}

    with ThreadPoolExecutor(max_workers=27) as pool:
        results = list(pool.map(execute_run, jobs))
    save_json(destination / "results.json", results)
    infer_positions(results, destination)


def infer_positions(results, destination, horizon=10):
    branches = ("TH", "TL", "U0")
    groups = sorted({(row["graph"], row["repeat"]) for row in results})
    lookup = {(row["graph"], row["repeat"], row["branch"]): row for row in results}
    values = np.array([[lookup[(*group, branch)]["summary"]["perceived_majority"] for branch in branches]
                       for group in groups])
    time_values = np.array([[
        np.mean([m["perceived_majority"] for m in lookup[(*group, branch)]["metrics"][horizon-3:horizon]])
        - np.mean([m["perceived_majority"] for m in lookup[(*group, branch)]["metrics"][2:5]])
        for branch in branches] for group in groups])
    graph_ids = np.array([group[0] for group in groups])
    rng = np.random.default_rng(19001)
    resampled = []
    time_resampled = []
    for _ in range(10000):
        indexes = np.concatenate([rng.choice(np.flatnonzero(graph_ids == graph),
                                              size=sum(graph_ids == graph), replace=True) for graph in range(3)])
        resampled.append(values[indexes].mean(axis=0))
        time_resampled.append(time_values[indexes].mean(axis=0))
    bootstrap = np.array(resampled)
    time_bootstrap = np.array(time_resampled)
    signs = np.array(list(itertools.product((-1, 1), repeat=len(groups))))
    comparisons = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        differences = values[:, first] - values[:, second]
        difference = float(differences.mean())
        null = (signs * differences).mean(axis=1)
        pvalue = float(np.mean(abs(null) >= abs(difference) - 1e-12))
        interval = np.quantile(bootstrap[:, first] - bootstrap[:, second], [0.025, 0.975])
        comparisons.append({"branches": [branches[first], branches[second]], "difference": difference,
            "ci95": interval.tolist(), "p_value": pvalue, "paired_groups": len(groups)})
    previous = 0.0
    for rank, index in enumerate(sorted(range(3), key=lambda i: comparisons[i]["p_value"])):
        previous = max(previous, min(1.0, comparisons[index]["p_value"] * (3 - rank)))
        comparisons[index]["p_holm"] = previous
    summaries = {branch: {"mean": float(values[:, index].mean()),
                          "ci95": np.quantile(bootstrap[:, index], [0.025, 0.975]).tolist()}
                 for index, branch in enumerate(branches)}
    winner = max(summaries, key=lambda branch: summaries[branch]["mean"])
    time_comparisons = []
    for index, branch in enumerate(branches):
        difference = float(time_values[:, index].mean())
        null = (signs * time_values[:, index]).mean(axis=1)
        time_comparisons.append({"branch": branch, "difference": difference,
            "ci95": np.quantile(time_bootstrap[:, index], [0.025, 0.975]).tolist(),
            "p_value": float(np.mean(abs(null) >= abs(difference) - 1e-12))})
    previous = 0.0
    for rank, index in enumerate(sorted(range(3), key=lambda i: time_comparisons[i]["p_value"])):
        previous = max(previous, min(1.0, time_comparisons[index]["p_value"] * (3 - rank)))
        time_comparisons[index]["p_holm"] = previous
    save_json(destination / "inference.json", {"branch_summaries": summaries, "comparisons": comparisons,
        "time_comparisons": time_comparisons,
        "recommended_position": winner, "primary_outcome": "tail observer majority illusion",
        "sign_flip_assumption": "paired differences are sign-exchangeable under the null, conditional on fixed networks"})


def main():
    global ROOT, INITIALIZATION
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("texts", "texts50", "screen", "confirm", "initialize", "controls", "refine", "freeze", "positions", "main"))
    parser.add_argument("--output", type=Path, default=ARCHIVE, help="Experiment output directory")
    args = parser.parse_args()
    ROOT = args.output.resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    INITIALIZATION = ROOT / "02_initialization_50"
    if args.stage == "texts":
        expand_bank()
    elif args.stage == "texts50":
        expand_bank(25)
    elif args.stage == "screen":
        screening()
    elif args.stage == "confirm":
        confirm_initialization()
    elif args.stage == "initialize":
        screening()
        confirm_initialization()
    elif args.stage == "positions":
        position_experiment()
    elif args.stage == "controls":
        explore_controls()
    elif args.stage == "refine":
        refine_initialization()
    elif args.stage == "freeze":
        freeze_initialization()
    elif args.stage == "main":
        freeze_initialization()
        position_experiment()


if __name__ == "__main__":
    main()
