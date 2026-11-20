import gzip
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from examples.demo import DemoPerception, demo_inputs
from majority.models import Action, Coefficients, RunConfig
from majority.randomness import RandomStreams
from majority.simulation import Simulation
from scripts.ext_common import parse_attitude
from scripts.ext_resistance import extract_conversions


def flip_perception():
    class Perception(DemoPerception):
        def estimate(self, request):
            result = super().estimate(request)
            return replace(result, new_stance=-int(request.state.stance))
    return Perception()


class PrivateUpdateTests(unittest.TestCase):
    def setUp(self):
        self.network, self.profiles, self.bank = demo_inputs()
        self.coefficients = Coefficients((0, 0, 0, 0), 1, 1, beta=2)

    def simulation(self, config, perception):
        return Simulation(self.network, self.profiles, frozenset({0}), self.bank,
                          self.coefficients, config, perception, RandomStreams(42), "TH")

    def test_update_off_ignores_new_stance(self):
        """R0 坚持条件：解析得到 new_stance 但 private_update=False，运行不报错、立场不变。"""
        simulation = self.simulation(RunConfig(rounds=3, private_update=False), flip_perception())
        for record in simulation.run():
            self.assertTrue(all(row.after.stance == row.before.stance for row in record.nodes))

    def test_update_on_commit_timing(self):
        """本轮发布消息必须与 before 立场一致；翻转观察者的下一轮 POST 决策使用更新后立场。"""
        simulation = self.simulation(RunConfig(rounds=3, private_update=True), flip_perception())
        cases = 0
        for record in simulation.run():
            rows = {row.node: row for row in record.nodes}
            for message in record.published:
                row = rows[message.author]
                if message.action == Action.POST:
                    self.assertEqual(int(message.stance), int(row.before.stance),
                                     "新立场不得与本轮消息一起提交")
            for row in record.nodes:
                if row.node in self.network.observers and row.after.stance != row.before.stance:
                    if row.next_decision.action == Action.POST:
                        self.assertEqual(row.next_decision.public_stance, int(row.after.stance),
                                         "下一轮决策必须使用更新后的私人立场")
                        cases += 1
        self.assertGreater(cases, 0, "测试场景未产生翻转观察者的原创决策，fixture 失效")

    def test_advocate_never_updates(self):
        """倡议者（节点0）即使收到 new_stance 也保持固定立场。"""
        simulation = self.simulation(RunConfig(rounds=3, private_update=True), flip_perception())
        for record in simulation.run():
            for row in record.nodes:
                if row.node == 0:
                    self.assertEqual(row.after.stance, row.before.stance)
                self.assertNotEqual(row.after.stance.value, 0)

    def test_invalid_new_stance_raises(self):
        class Perception(DemoPerception):
            def estimate(self, request):
                return replace(super().estimate(request), new_stance=0)
        simulation = self.simulation(RunConfig(rounds=3, private_update=True), Perception())
        with self.assertRaises(ValueError):
            next(simulation.run())

    def test_bool_new_stance_raises(self):
        class Perception(DemoPerception):
            def estimate(self, request):
                return replace(super().estimate(request), new_stance=True)
        simulation = self.simulation(RunConfig(rounds=3, private_update=True), Perception())
        with self.assertRaises(ValueError):
            next(simulation.run())

    def test_missing_new_stance_raises(self):
        """private_update=True 而感知结果缺少 new_stance（如误用 v1 提示词）必须报错，不静默。"""
        simulation = self.simulation(RunConfig(rounds=3, private_update=True), DemoPerception())
        with self.assertRaises(ValueError):
            next(simulation.run())


class AttitudeParseTests(unittest.TestCase):
    def test_valid_payload(self):
        support, reason, ids, stance = parse_attitude(
            {"support": 0.3, "reason": "所见反对更多", "evidence_ids": [], "new_stance": -1}, set())
        self.assertEqual((support, reason, ids, stance), (0.3, "所见反对更多", (), -1))

    def test_rejects_invalid_payloads(self):
        for data in ({"support": 0.3, "reason": "x", "evidence_ids": []},
                     {"support": 0.3, "reason": "x", "evidence_ids": [], "new_stance": 0},
                     {"support": 0.3, "reason": "x", "evidence_ids": [], "new_stance": True},
                     {"support": 0.3, "reason": "x", "evidence_ids": [], "new_stance": 1.0},
                     {"support": 0.3, "reason": "x", "evidence_ids": [], "new_stance": 1, "extra": 1}):
            with self.assertRaises(ValueError):
                parse_attitude(data, set())

    def test_rejects_unknown_evidence(self):
        with self.assertRaises(ValueError):
            parse_attitude({"support": 0.3, "reason": "x", "evidence_ids": ["m9"], "new_stance": 1},
                           {"m1"})


class ConversionsTests(unittest.TestCase):
    def test_extract_conversions_counts_flips_and_support(self):
        lines = []
        for round_number, stance_pairs in ((1, [(1, 1), (1, -1), (-1, -1)]),
                                           (2, [(-1, 1), (-1, -1), (1, 1)])):
            nodes = [{"node": i, "before": {"stance": before, "perceived_support": 0.5, "feedback_value": 0.0},
                      "after": {"stance": after, "perceived_support": 0.5, "feedback_value": 0.0}}
                     for i, (before, after) in enumerate(stance_pairs)]
            lines.append(json.dumps({"kind": "round",
                                     "data": {"round": round_number, "branch": "TH",
                                              "published": [], "nodes": nodes,
                                              "advocate_stance": -1}}))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                for line in lines:
                    stream.write(line + "\n")
            per_round = extract_conversions(path)
        self.assertEqual(per_round[1]["flipped_to_oppose"], 1)
        self.assertEqual(per_round[1]["flipped_to_support"], 0)
        self.assertAlmostEqual(per_round[1]["true_support"], 2 / 3)
        self.assertEqual(per_round[2]["flipped_to_oppose"], 0)
        self.assertEqual(per_round[2]["flipped_to_support"], 1)
        self.assertAlmostEqual(per_round[2]["true_support"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
