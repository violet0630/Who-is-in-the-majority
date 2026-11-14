import io
import json
import unittest
from dataclasses import asdict, replace
from math import nextafter
from random import Random

from examples.demo import DemoPerception, demo_inputs
from majority.analysis import measure
from majority.behavior import decide, update_feedback
from majority.experiments import SeedSelection, plan_group, run_group
from majority.feed import make_message, read_feed
from majority.models import (
    Action, ActionProbabilities, AgentState, Coefficients, Decision, Message,
    Network, Profile, RunConfig, Stance, Text,
)
from majority.perception import PerceptionResult
from majority.randomness import RandomStreams
from majority.simulation import Simulation
from majority.storage import write_record


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.network, self.profiles, self.bank = demo_inputs()
        self.coefficients = Coefficients((0, 0, 0, 0), 1, 1, beta=2)
        self.config = RunConfig(rounds=3, advocate_stance=Stance.SUPPORT)
        self.streams = RandomStreams(42)

    def original(self, id="original", author=0, round=1, stance=Stance.OPPOSE):
        template = next(item for item in self.bank.originals if item.stance == stance)
        return Message(id, author, round, Action.POST, stance, template.text,
                       None, frozenset({4}))

    def simulation(self):
        return Simulation(self.network, self.profiles, frozenset({0}), self.bank,
                          self.coefficients, self.config, DemoPerception(), self.streams, "TH")

    def test_feedback_and_speech_odds(self):
        state = AgentState(Stance.OPPOSE)
        updated = update_feedback(state, 3, 1, 0.2)
        self.assertAlmostEqual(updated.feedback_value, 0.08)
        self.assertAlmostEqual(update_feedback(updated, 0, 0, 0.2).feedback_value, 0.064)
        for advocate, rho, expected in ((False, 6, 0.2), (True, 3, 3 / 7), (True, 6, 0.6)):
            decision = decide(0, 1, Profile(0.2, 0.4), state, advocate, (),
                              self.coefficients, replace(self.config, rho=rho), self.streams)
            self.assertAlmostEqual(decision.probabilities.post, expected)
            self.assertEqual(decision.probabilities.agree + decision.probabilities.disagree, 0)
            self.assertAlmostEqual(sum(asdict(decision.probabilities).values()), 1)

    def test_advocates_publicly_support_both_target_stances(self):
        coefficients = Coefficients((1000, 0, 0, 0), 1, 1)
        for stance in Stance:
            decision = decide(
                0, 1, Profile(nextafter(1.0, 0.0), 0.8), AgentState(Stance.SUPPORT),
                True, (self.original(stance=stance),), coefficients, self.config, self.streams,
            )
            expected = Action.AGREE if stance == Stance.SUPPORT else Action.DISAGREE
            self.assertEqual(decision.action, expected)
            self.assertEqual(decision.public_stance, 1)

    def test_context_deduplication_age_and_own_posts(self):
        current = self.original()
        expired = self.original("expired", round=0)
        own = self.original("own", author=4)
        comments = tuple(
            Message(f"reply-{i}", 2, 2, Action.AGREE, Stance.OPPOSE,
                    Text(f"comment-{i}", "赞同"), target, frozenset({4}))
            for i, target in enumerate((current.id, current.id, expired.id))
        )
        messages = {m.id: m for m in (current, expired, own, *comments)}
        reading = read_feed(4, 2, messages, replace(self.config, read_limit=10), Random(1))
        self.assertEqual(reading.context, ("expired", "original"))
        self.assertEqual(reading.targets, ("original",))
        self.assertEqual(len(reading.read), 5)
        self.assertNotIn("expired", reading.visible)
        reading = read_feed(4, 2, messages, self.config, Random(1))
        self.assertLessEqual(len(reading.read), 5)

    def test_comment_audience_and_public_stance(self):
        original = self.original()
        decision = Decision(Action.DISAGREE, ActionProbabilities(0, 1, 0, 0), 1, original.id)
        for cross in (False, True):
            message = make_message("TH", 2, 2, decision, self.network, self.bank,
                                   {original.id: original},
                                   replace(self.config, cross_fan_comments=cross), Random(1))
            expected = self.network.fans[2]
            if cross:
                expected |= self.network.fans[0]
            self.assertEqual(message.audience, expected - {2})
            self.assertEqual(message.stance, Stance.SUPPORT)
            self.assertIn("[支持]", message.text.body)

    def test_synchronous_publication_and_notifications_once(self):
        self.network = Network((0,) * 8, tuple(frozenset(set(range(8)) - {i}) for i in range(8)),
                               frozenset({4, 5, 6, 7}))
        self.profiles = (Profile(nextafter(1.0, 0.0), 0.4),) * 8
        self.coefficients = Coefficients((1000, 0, 0, 0), 0, 0)
        simulation = self.simulation()
        first, second, third = tuple(simulation.run())
        self.assertTrue(all(m.action == Action.POST and m.round == 1 for m in first.published))
        self.assertTrue(all(not row.notifications for row in first.nodes))
        self.assertTrue(all(row.next_decision.target_id for row in first.nodes))
        self.assertTrue(all(m.round == 2 and m.original_id for m in second.published))
        second_ids = {m.id for m in second.published}
        notices = [id for row in second.nodes for id in row.notifications]
        self.assertEqual(set(notices), second_ids)
        self.assertEqual(len(notices), len(second_ids))
        self.assertFalse(second_ids & {id for row in third.nodes for id in row.notifications})
        self.assertTrue(all(simulation.messages[id].round <= 1
                            for row in first.nodes for id in row.reading.read))
        for row in second.nodes:
            replies = [simulation.messages[id] for id in row.notifications]
            agrees = sum(m.action == Action.AGREE for m in replies)
            expected = self.config.alpha * (2 * agrees - len(replies)) / (1 + len(replies))
            self.assertAlmostEqual(row.after.feedback_value, expected)

    def test_empty_reading_retains_perception_and_metrics_denominators(self):
        simulation = self.simulation()
        simulation.pending = ()
        simulation.states = tuple(replace(state, perceived_support=0.6) for state in simulation.states)
        record = simulation.step()
        self.assertTrue(all(row.perception is None and row.after.perceived_support == 0.6
                            for row in record.nodes))
        metrics = measure(record, self.network, simulation.messages)
        self.assertEqual(metrics.feed_majority, 0)
        self.assertEqual(metrics.perceived_majority, 1)
        self.assertEqual(metrics.public_support, 0)
        self.assertEqual(metrics.opponent_silence, 1)
        self.assertEqual(metrics.public_conformity, 0)
        self.assertEqual(metrics.empty_readers, 4)
        self.assertAlmostEqual(metrics.perception_bias, 0.6 - 1 / 8)

    def test_paired_streams_and_branch_isolation(self):
        seeds = SeedSelection(frozenset({0}), frozenset({0}), frozenset({0}), 1, 0)
        specs = plan_group("graph", "profiles", seeds, 42)
        runs = list(run_group(specs, self.network, self.profiles, self.bank,
                              self.coefficients, self.config, DemoPerception))
        signatures = [tuple(tuple(row.next_decision.action for row in record.nodes)
                            for record in records) for _, records in runs]
        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(signatures[0], signatures[2])
        self.assertNotEqual(RandomStreams(42, "paired").for_event(0, 1, "speak").random(),
                            RandomStreams(42, "independent").for_event(0, 1, "speak").random())
        a, b = self.simulation(), self.simulation()
        a.step()
        self.assertFalse(b.messages)
        self.assertTrue(all(not history for history in b.history))

    def test_external_boundaries_and_log_serialization(self):
        for value in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                PerceptionResult(value, "{}")
        with self.assertRaises(ValueError):
            Simulation(self.network, self.profiles, frozenset({4}), self.bank,
                       self.coefficients, self.config, DemoPerception(), self.streams, "bad")
        buffer = io.StringIO()
        write_record(buffer, "round", self.simulation().step())
        value = json.loads(buffer.getvalue())
        self.assertEqual(value["data"]["round"], 1)
        self.assertEqual(value["kind"], "round")


if __name__ == "__main__":
    unittest.main()
