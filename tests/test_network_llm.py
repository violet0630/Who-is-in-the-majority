import io
import json
import tempfile
import unittest
from collections import Counter
from math import nextafter
from pathlib import Path
from random import Random

import httpx
from openai import OpenAI

from examples.demo import demo_inputs
from majority.analysis import measure, tail_illusion
from majority.judgments import summarize_calls, summarize_judgments
from majority.llm import JsonChat, LLMConfig, LLMError
from majority.llm_perception import LLMPerception, perception_payload
from majority.models import AgentState, Coefficients, Network, Profile, RunConfig, Stance
from majority.network import NetworkConfig, generate_network, network_statistics
from majority.perception import Exposure, PerceptionRequest, PerceptionResult, ViewedMessage
from majority.randomness import RandomStreams
from majority.simulation import Simulation


class NetworkTests(unittest.TestCase):
    def test_reproducible_constrained_network_and_observer_strata(self):
        config = NetworkConfig()
        network = generate_network(config, Random(42))
        self.assertEqual(network, generate_network(config, Random(42)))
        self.assertEqual(sorted(Counter(network.communities).values()), [15, 20, 25, 40])
        stats = network_statistics(network)
        self.assertTrue(760 <= stats.edges <= 840)
        self.assertTrue(all(stats.following_counts))
        self.assertEqual(len(network.observers), 30)
        for node, followed in enumerate(network.following):
            self.assertNotIn(node, followed)
            self.assertTrue(all(node in network.fans[target] for target in followed))
        seen, pending = {0}, [0]
        while pending:
            node = pending.pop()
            for neighbor in (network.following[node] | network.fans[node]) - seen:
                seen.add(neighbor)
                pending.append(neighbor)
        self.assertEqual(len(seen), 100)
        for group in stats.observer_strata:
            observed = len(set(group) & network.observers)
            self.assertLessEqual(abs(observed - len(group) * 0.3), 1)

    def test_statistics_and_impossible_configuration(self):
        graph = Network((0, 0, 1), (frozenset({1}), frozenset({2}), frozenset({0})), frozenset({2}))
        stats = network_statistics(graph)
        self.assertEqual(stats.undirected_clustering, 1)
        self.assertEqual(stats.reciprocal_edge_fraction, 0)
        self.assertAlmostEqual(stats.cross_community_fraction, 2 / 3)
        with self.assertRaises(ValueError):
            NetworkConfig(between_weight=0)


class LLMTests(unittest.TestCase):
    def setUp(self):
        self.audit = io.StringIO()
        message = ViewedMessage("TH:1:0", 0, 1, "这是一条支持当选的测试消息。", None)
        self.request = PerceptionRequest(1, Profile(0.2, 0.8), AgentState(Stance.SUPPORT), (),
                                         Exposure(1, (message,), ()))
        self.valid = {"support": 0.3, "reason": "根据当前可见消息作出有限估计。", "evidence_ids": ["r1-u0"]}

    def client(self, content, *, finish="stop", usage=True, status=200):
        def respond(request):
            self.sent = json.loads(request.content)
            self.assertEqual(str(request.url), "http://127.0.0.1:3000/v1/chat/completions")
            data = {
                "id": "test-request", "object": "chat.completion", "created": 1,
                "model": "deepseek-v4-flash", "choices": [{"index": 0, "finish_reason": finish,
                    "message": {"role": "assistant", "content": json.dumps(content)}}],
            }
            if usage:
                data["usage"] = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            return httpx.Response(status, json=data)
        client = OpenAI(api_key="test-secret", base_url="http://127.0.0.1:3000/v1", max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        self.addCleanup(client.close)
        return JsonChat(client, LLMConfig(), self.audit)

    def test_schema_usage_and_information_boundary(self):
        model = LLMPerception(self.client(self.valid), "是否支持当选？", "TH", "2024 年投票前")
        result = model.estimate(self.request)
        self.assertEqual(result.support, 0.3)
        self.assertEqual((result.input_tokens, result.output_tokens), (10, 20))
        self.assertEqual(self.sent["temperature"], 0)
        self.assertEqual(self.sent["response_format"], {"type": "json_object"})
        payload = json.loads(self.sent["messages"][1]["content"])
        self.assertEqual(payload["scenario"], "2024 年投票前")
        self.assertNotIn("TH", json.dumps(payload))
        for forbidden in ("advocates", "audience", "true_support", "baseline", "feedback_value"):
            self.assertNotIn(forbidden, json.dumps(payload))
        self.assertNotIn("test-secret", self.audit.getvalue())

    def test_invalid_outputs_fail_and_remain_in_audit(self):
        for support in (True, -0.1, 1.1, float("nan")):
            model = LLMPerception(self.client({**self.valid, "support": support}), "议题", "test")
            with self.assertRaises(LLMError):
                model.estimate(self.request)
        unknown = {**self.valid, "evidence_ids": ["unseen"]}
        with self.assertRaises(LLMError):
            LLMPerception(self.client(unknown), "议题", "test").estimate(self.request)
        with self.assertRaises(LLMError):
            LLMPerception(self.client(self.valid, finish="length"), "议题", "test").estimate(self.request)
        rows = [json.loads(line)["data"] for line in self.audit.getvalue().splitlines()]
        self.assertTrue(all(row["status"] == "invalid_response" for row in rows))

    def test_api_failure_and_missing_usage_statistics(self):
        with self.assertRaises(LLMError):
            LLMPerception(self.client(self.valid, status=401), "议题", "test").estimate(self.request)
        result = LLMPerception(self.client(self.valid, usage=False), "议题", "test").estimate(self.request)
        self.assertIsNone(result.input_tokens)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calls.jsonl"
            path.write_text(self.audit.getvalue(), encoding="utf-8")
            summary = summarize_calls(path)["perception"]
        self.assertEqual(summary["calls"], 2)
        self.assertEqual(summary["api_errors"], 1)
        self.assertEqual(summary["successful"], 1)
        self.assertEqual(summary["calls_without_usage"], 2)

    def test_reference_context_and_branch_aliases(self):
        original = self.request.current.messages[0]
        reply = ViewedMessage("TH:2:3", 3, 2, "我反对原帖。", original.id)
        request = PerceptionRequest(1, self.request.profile, self.request.state,
                                    (self.request.current,), Exposure(2, (reply,), (original,)))
        payload = perception_payload("议题", request)
        self.assertEqual(payload["current"]["messages"][0]["original_id"], "r1-u0")
        self.assertEqual(len(payload["current"]["messages"]), 1)
        self.assertEqual(len(payload["current"]["context"]), 1)


class OpposingAdvocatesTests(unittest.TestCase):
    def test_opposing_minority_and_reversed_illusion_metrics(self):
        _, _, bank = demo_inputs()
        network = Network((0,) * 4, tuple(frozenset(set(range(4)) - {i}) for i in range(4)),
                          frozenset({1, 2, 3}))

        class Perception:
            def estimate(self, request):
                return PerceptionResult(0.1, '{"support": 0.1}')

        sim = Simulation(network, (Profile(nextafter(1.0, 0.0), 0.8),) * 4, frozenset({0}), bank,
                         Coefficients((1000, 0, 0, 0), 0, 1000), RunConfig(rounds=2),
                         Perception(), RandomStreams(42), "U0")
        self.assertEqual([state.stance for state in sim.states], [-1, 1, 1, 1])
        first, second = tuple(sim.run())
        for record in (first, second):
            self.assertEqual(next(m.stance for m in record.published if m.author == 0), -1)
        metrics = measure(second, network, sim.messages)
        self.assertEqual(metrics.true_support, 0.75)
        self.assertEqual(metrics.true_advocate_support, 0.25)
        self.assertEqual(metrics.perceived_majority, 1)
        self.assertEqual(metrics.public_conformity, 1)
        self.assertEqual(metrics.public_support, 0)
        self.assertAlmostEqual(metrics.advocate_perception_bias, 0.65)
        self.assertEqual(tail_illusion([metrics], last=1), 1)
        judgments = summarize_judgments(second, network)
        self.assertEqual(judgments.support_majority_count, 0)
        self.assertEqual(judgments.advocate_majority_count, 3)
        self.assertEqual(judgments.updated, 3)


if __name__ == "__main__":
    unittest.main()
