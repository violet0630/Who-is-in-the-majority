import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from examples.demo import demo_inputs
from majority.analysis import measure
from majority.llm_perception import perception_payload
from majority.models import Action, AgentState, Message, Network, Reading, Stance, message_order
from majority.perception import Exposure, PerceptionRequest, ViewedMessage
from scripts.majority_factorial import PROJECT, build_tasks, read_json, task_settings, validate_protocol
from scripts.majority_study import ObserverPerception, run_one
from scripts.summarize_factorial import analyze_rows, plot_results


class V2Tests(unittest.TestCase):
    def setUp(self):
        self.net, self.profiles, self.bank = demo_inputs()
        self.positive = next(x.text for x in self.bank.originals if x.stance == Stance.SUPPORT)
        self.negative = next(x.text for x in self.bank.originals if x.stance == Stance.OPPOSE)

    def test_latest_author_numeric_round_and_slot_and_no_context_vote(self):
        perception = ObserverPerception(frozenset({99}), self.bank)
        seen = (ViewedMessage('TH:10:1', 1, 10, self.positive.body, None),
                ViewedMessage('TH:9:1', 1, 9, self.negative.body, None),
                ViewedMessage('TH:10:1:c2', 1, 10, self.negative.body, None),
                ViewedMessage('TH:10:1:c1', 1, 10, self.positive.body, None),
                ViewedMessage('TH:10:2', 2, 10, self.positive.body, None))
        context = (ViewedMessage('TH:10:3', 3, 10, self.negative.body, None),)
        req = PerceptionRequest(0, self.profiles[0], AgentState(Stance.SUPPORT, .73), (), Exposure(10, seen, context))
        self.assertEqual(perception.estimate(req).support, .5)
        self.assertEqual(perception.estimate(replace(req, current=Exposure(10, tuple(reversed(seen)), context))).support, .5)
        self.assertEqual(perception.estimate(replace(req, current=Exposure(10, (), context))).support, .73)

    def test_multicomment_aliases_unique_and_branch_neutral(self):
        def payload(branch):
            original = ViewedMessage(f'{branch}:9:2', 2, 9, self.positive.body, None)
            messages = tuple(ViewedMessage(f'{branch}:10:1' + (f':c{i}' if i else ''),
                             1, 10, self.positive.body, original.id) for i in range(3))
            req = PerceptionRequest(0, self.profiles[0], AgentState(Stance.SUPPORT), (), Exposure(10, messages, (original,)))
            return perception_payload('topic', req)
        data = payload('TH')
        self.assertEqual(data, payload('TL'))
        self.assertEqual(len({m['id'] for m in data['current']['messages']}), 3)
        self.assertTrue(all(m['original_id'] == 'r9-u2' for m in data['current']['messages']))

    def test_latest_public_stance_independent_of_storage_order(self):
        network = Network((0, 0, 0), (frozenset({1}), frozenset({2}), frozenset({0})), frozenset({0}))
        rows = tuple(SimpleNamespace(node=i, before=AgentState(Stance.SUPPORT if i != 1 else Stance.OPPOSE),
                 after=AgentState(Stance.SUPPORT if i != 1 else Stance.OPPOSE), reading=Reading()) for i in range(3))
        messages = (Message('TH:1:0', 0, 1, Action.AGREE, Stance.SUPPORT, self.positive, 'x', frozenset()),
                    Message('TH:1:0:c1', 0, 1, Action.DISAGREE, Stance.OPPOSE, self.negative, 'y', frozenset()))
        values = [measure(SimpleNamespace(nodes=rows, advocate_stance=Stance.OPPOSE, published=order, round=1), network, {})
                  for order in (messages, tuple(reversed(messages)))]
        self.assertEqual(values[0], values[1])
        self.assertEqual(values[0].public_conformity, 1)

    def test_v2_grid_and_actual_config(self):
        protocol = read_json(PROJECT / 'configs/main2_factorial_v2.json')
        baseline = read_json(PROJECT / protocol['baseline'])
        validate_protocol(protocol, baseline)
        groups = [{'graph': g, 'repeat': 0, 'network_file': 'net', 'profiles_file': 'profiles',
                   'positions': {str(k): str(k) for k in protocol['k_grid']}} for g in protocol['graphs']]
        tasks = build_tasks(protocol, groups)
        self.assertEqual(len(tasks), 36)
        self.assertEqual(sum(t['pilot'] for t in tasks), 9)
        settings = task_settings(baseline, protocol, tasks[0])
        captured = []
        def capture(*args, **kwargs):
            captured.append(args[5])
            raise RuntimeError('capture')
        with patch('scripts.majority_study.ObserverSimulation', capture):
            with self.assertRaisesRegex(RuntimeError, 'capture'):
                run_one(self.net, self.profiles, frozenset({0}), self.bank,
                        {**settings, 'max_comments': 2, 'comment_alpha': 2, 'comment_beta': 5}, 42, 'TH')
        self.assertEqual(captured[0].read_limit, 11)
        self.assertEqual((captured[0].max_comments, captured[0].comment_alpha, captured[0].comment_beta), (2, 2, 5))

    def test_three_graph_analysis_and_plots(self):
        protocol = read_json(PROJECT / 'configs/main2_factorial_v2.json')
        rows = []
        for g in protocol['graphs']:
            for x, k in enumerate(protocol['k_grid']):
                for y, rho in enumerate(protocol['rho_grid']):
                    q = .1 + .02*g + .03*x + .04*y + .01*x*y
                    rows.append({'graph': g, 'repeat': 0, 'k': k, 'rho': rho, 'q_tail': q,
                                 'metrics': [{'perceived_majority': q, 'feed_majority': q} for _ in range(10)]})
        analysis = analyze_rows(rows, protocol)
        self.assertEqual(analysis['paired_groups'], 3)
        self.assertNotIn('ci', analysis['cells'][0])
        for item, expected in zip(analysis['comparisons'], (.12, .11, .06)):
            self.assertAlmostEqual(item['difference'], expected)
            self.assertNotIn('p_value', item)
        with self.assertRaises(ValueError):
            analyze_rows(rows[:-1], protocol)
        with TemporaryDirectory() as directory:
            plot_results(rows, analysis, Path(directory), protocol)
            self.assertEqual(len(list(Path(directory).glob('*.png'))), 4)


if __name__ == '__main__':
    unittest.main()
