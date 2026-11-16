import json
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
import unittest

from scripts.majority_study import (
    ObserverPerception, ObserverSimulation, coefficients_from, infer_positions,
    make_network, select_positions,
)
from majority.models import Action, AgentState, Message, RunConfig, Stance
from majority.perception import Exposure, PerceptionRequest, PerceptionResult, ViewedMessage
from majority.preparation import draw_profiles
from majority.randomness import RandomStreams
from majority.artifacts import load_bank
from scripts.majority_study import active_bank


SETTINGS = {"mean_degree": 8, "s_out": 0.35, "s_in": 1, "w": 0.2,
            "kappa": 1, "gamma": 2, "theta": [0, 0.5, 1, 1]}


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network = make_network(100, SETTINGS, 123)
        cls.profiles = draw_profiles(100, Random(124))
        _, cls.bank = load_bank(active_bank())
        cls.positions, cls.diagnostics = select_positions(cls.network, cls.profiles, 16, 125, steps=300, restarts=2)

    def test_only_observers_call_llm(self):
        class Spy:
            calls = []
            def estimate(self, request):
                self.calls.append(request.node)
                return PerceptionResult(0.7, "{}")
        spy = Spy()
        perception = ObserverPerception(self.network.observers, self.bank, spy)
        for node in range(100):
            request = PerceptionRequest(node, self.profiles[node], AgentState(Stance.SUPPORT), (), Exposure(1, (), ()))
            result = perception.estimate(request)
            self.assertEqual(result.support, 0.7 if node in self.network.observers else 0.5)
        self.assertEqual(set(spy.calls), set(self.network.observers))
        self.assertEqual(len(spy.calls), 50)

    def test_nonobserver_decisions_ignore_perceived_climate(self):
        simulation = ObserverSimulation(self.network, self.profiles, self.positions["U0"], self.bank,
            coefficients_from(SETTINGS), RunConfig(), ObserverPerception(self.network.observers, self.bank),
            RandomStreams(126, "paired"), "U0")
        node = next(i for i in range(100) if i not in self.network.observers)
        text = next(item.text for item in self.bank.originals if item.stance == Stance.SUPPORT)
        simulation.messages["m"] = Message("m", (node + 1) % 100, 1, Action.POST, Stance.SUPPORT,
                                            text, None, frozenset({node}))
        first = simulation._decide(node, 2, AgentState(Stance.SUPPORT, 0.1, 0.2), ("m",))
        second = simulation._decide(node, 2, AgentState(Stance.SUPPORT, 0.9, 0.2), ("m",))
        self.assertEqual(first, second)

    def test_position_matching_and_constraints(self):
        self.assertEqual(len(self.network.observers), 50)
        self.assertTrue(760 <= sum(map(len, self.network.following)) <= 840)
        for selected in self.positions.values():
            self.assertEqual(len(selected), 16)
            self.assertFalse(selected & self.network.observers)
        for field in ("fan_total", "baseline_total"):
            high, low = self.diagnostics["TH"][field], self.diagnostics["TL"][field]
            self.assertLessEqual(2 * abs(high - low) / (high + low), 0.05)
        for field in ("advocate_strong_count",):
            self.assertEqual(self.diagnostics["TH"][field], self.diagnostics["TL"][field])

    def test_run_level_inference(self):
        rows = [{"graph": graph, "repeat": repeat, "branch": branch,
                 "summary": {"perceived_majority": value},
                 "metrics": [{"perceived_majority": value} for _ in range(10)]}
                for graph in range(3) for repeat in range(3)
                for branch, value in (("TH", 0.4), ("TL", 0.1), ("U0", 0.2))]
        with TemporaryDirectory() as directory:
            destination = Path(directory)
            infer_positions(rows, destination)
            result = json.loads((destination / "inference.json").read_text(encoding="utf-8"))
        self.assertEqual(result["recommended_position"], "TH")
        primary = result["comparisons"][0]
        self.assertAlmostEqual(primary["difference"], 0.3)
        self.assertAlmostEqual(primary["p_value"], 2 / 512)
        self.assertAlmostEqual(primary["p_holm"], 6 / 512)
        self.assertTrue(all(row["p_value"] == 1 for row in result["time_comparisons"]))


if __name__ == "__main__":
    unittest.main()
