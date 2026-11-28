import gzip
import io
import json
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
from openai import OpenAI

from examples.demo import demo_inputs
from majority.llm import JsonChat, LLMConfig, LLMValidationError, open_client
from majority.models import Action, AgentState, Coefficients, Message, RunConfig, Stance
from majority.perception import PerceptionResult
from majority.randomness import RandomStreams
from majority.simulation import Simulation
from scripts.main2_v2 import audit_trajectory
from scripts.majority_factorial import PROJECT, build_tasks, read_json, task_settings, validate_protocol
from scripts.majority_study import ObserverPerception, StudyChat, run_one, study_llm_config


class ObserverMeanTests(unittest.TestCase):
    def simulation(self, spy, workers=1):
        net, profiles, bank = demo_inputs()
        config = RunConfig(rounds=2, nonobserver_perception="observer_mean")
        sim = Simulation(net, profiles, frozenset({0}), bank,
                         Coefficients((1, 0, -1, 0), .5, 1), config,
                         ObserverPerception(net.observers, bank, spy),
                         RandomStreams(123, "paired"), "TH", workers=workers)
        negative = next(o.text for o in bank.originals if o.stance == -1)
        sim.pending = (Message("TH:1:0", 0, 1, Action.POST, Stance.OPPOSE,
                               negative, None, frozenset({1, 4, 5, 6})),)
        sim.states = tuple(replace(s, perceived_support=.9) if i == 7 else s
                           for i, s in enumerate(sim.states))
        return sim

    def test_full_observer_mean_includes_empty_observers_and_all_nonobservers(self):
        spy = Mock()
        spy.estimate.side_effect = lambda req: PerceptionResult({4: .2, 5: .4, 6: .6}[req.node], "{}", model="fake-llm")
        sim = self.simulation(spy)
        original_pending = sim.pending
        with patch.object(sim, "_decide", wraps=sim._decide) as decisions:
            record = sim.step()
        self.assertEqual(record.published, original_pending)
        self.assertEqual(sorted(c.args[0].node for c in spy.estimate.call_args_list), [4, 5, 6])
        for row in record.nodes[:4]:
            self.assertAlmostEqual(row.after.perceived_support, .525)
            self.assertEqual(row.perception.model, "observer_mean")
            source = json.loads(row.perception.raw_json)
            self.assertEqual((source['source_round'], source['observer_count'], source['updated_observers']), (1, 4, 3))
            self.assertEqual(row.before.perceived_support, .5)
            self.assertEqual(row.after.stance, row.before.stance)
        self.assertEqual(record.nodes[7].after.perceived_support, .9)
        self.assertIsNone(record.nodes[7].perception)
        self.assertTrue(record.nodes[1].reading.read)
        self.assertFalse(record.nodes[2].reading.read)
        self.assertFalse(record.nodes[2].next_decision.comments)
        for call in decisions.call_args_list:
            node, round_number, state, targets = call.args
            self.assertEqual(round_number, 2)
            self.assertEqual(state, record.nodes[node].after)
            self.assertEqual(targets, record.nodes[node].reading.targets)

    def test_empty_round_keeps_fixed_denominator_and_makes_no_llm_calls(self):
        spy = Mock()
        sim = self.simulation(spy)
        sim.pending = ()
        record = sim.step()
        spy.estimate.assert_not_called()
        for row in record.nodes[:4]:
            self.assertAlmostEqual(row.after.perceived_support, .6)
        self.assertEqual(record.nodes[7].after.perceived_support, .9)

    def test_parallel_estimation_has_same_states_and_decisions(self):
        records = []
        for workers in (1, 4):
            spy = Mock()
            spy.estimate.side_effect = lambda req: PerceptionResult(req.node / 10, "{}", model="fake-llm")
            records.append(self.simulation(spy, workers).step())
        self.assertEqual(records[0], records[1])

    def test_failure_does_not_commit_partial_mean_or_next_round(self):
        spy = Mock()
        spy.estimate.side_effect = [PerceptionResult(.1, "{}"), RuntimeError("test failure")]
        sim = self.simulation(spy)
        before = sim.states
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            sim.step()
        self.assertEqual(sim.states, before)
        self.assertEqual(sim.next_round, 1)

    def test_protocol_grid_and_runner_select_actual_broadcast_config(self):
        protocol = read_json(PROJECT / "configs/main2_observer_mean.json")
        baseline = read_json(PROJECT / protocol["baseline"])
        validate_protocol(protocol, baseline)
        groups = [{"graph": g, "repeat": 0, "network_file": "net", "profiles_file": "profiles",
                   "positions": {str(k): str(k) for k in protocol["k_grid"]}} for g in range(3)]
        tasks = build_tasks(protocol, groups)
        self.assertEqual((len(tasks), sum(t['pilot'] for t in tasks)), (36, 9))
        settings = task_settings(baseline, protocol, tasks[0])
        captured = []
        def capture(*args, **kwargs):
            captured.append(args[5])
            raise RuntimeError("captured")
        net, profiles, bank = demo_inputs()
        with patch('scripts.majority_study.ObserverSimulation', capture):
            with self.assertRaisesRegex(RuntimeError, 'captured'):
                run_one(net, profiles, frozenset({0}), bank, settings, 1, 'TH')
        self.assertEqual(captured[0].nonobserver_perception, 'observer_mean')
        self.assertEqual((captured[0].read_limit, captured[0].rounds), (11, 10))
        self.assertEqual(study_llm_config(LLMConfig(**protocol['llm'])).timeout, 180)
        with self.assertRaises(ValueError):
            RunConfig(nonobserver_perception="typo")

    def test_independent_audit_rejects_wrong_average_and_source_round(self):
        config = asdict(RunConfig(rounds=1, nonobserver_perception='observer_mean'))
        state = lambda i: {'stance': -1 if i == 0 else 1, 'perceived_support': .5, 'feedback_value': 0}
        rows = []
        for i in range(1000):
            result = None if i >= 950 else asdict(PerceptionResult(.5, json.dumps({
                'source': 'observer_mean', 'source_round': 1, 'observer_count': 50,
                'updated_observers': 0}), model='observer_mean'))
            rows.append({'node': i, 'before': state(i), 'after': state(i), 'perception': result,
                         'reading': {'read': [], 'visible': []}, 'notifications': []})
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'trajectory.jsonl.gz'
            def audit():
                with gzip.open(path, 'wt', encoding='utf-8') as f:
                    for kind, data in [('manifest', {'config': config}),
                                       ('round', {'round': 1, 'published': [], 'nodes': rows})]:
                        f.write(json.dumps({'kind': kind, 'data': data})+'\n')
                return audit_trajectory(path, config, set(range(950, 1000)), {0}, expected_rounds=1)
            self.assertEqual(audit()['rounds'], 1)
            rows[2]['after']['perceived_support'] = .4
            with self.assertRaisesRegex(ValueError, 'broadcast differs'):
                audit()
            rows[2]['after']['perceived_support'] = .5
            rows[2]['perception']['raw_json'] = json.dumps({'source_round': 0})
            with self.assertRaisesRegex(ValueError, 'provenance differs'):
                audit()


class LocalTransportTests(unittest.TestCase):
    config = LLMConfig(base_url='http://127.0.0.1:3425/v1', model='qoder-cn/dfmodel',
                       timeout=180, max_tokens=4096, reasoning_effort='none')

    def test_full_runner_mocked_api_logs_new_mode_and_keeps_observer_history(self):
        net, profiles, bank = demo_inputs()
        requests = []
        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            return httpx.Response(200, json={'id': 'offline-test', 'object': 'chat.completion', 'created': 1,
                'model': 'qoder-cn/dfmodel', 'choices': [{'index': 0, 'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': '{"support": 0.3, "reason": "test", "evidence_ids": []}'}}]})
        settings = {'rounds': 2, 'rho': 3, 'kappa': .5, 'gamma': 1, 'theta': [1, 0, -1, 0],
                    'llm_workers': 2, 'nonobserver_perception': 'observer_mean'}
        with TemporaryDirectory() as folder:
            output = Path(folder) / 'run'
            client = OpenAI(api_key='test', base_url=self.config.base_url, timeout=180, max_retries=0,
                            http_client=httpx.Client(transport=httpx.MockTransport(respond)))
            with patch('scripts.majority_study.load_settings', return_value=(self.config, 'test')):
                with patch('scripts.majority_study.open_client', return_value=client):
                    run_one(net, profiles, frozenset({0}), bank, settings, 12, 'TH', output, True)
            with gzip.open(output / 'trajectory.jsonl.gz', 'rt', encoding='utf-8') as f:
                rows = [json.loads(line) for line in f]
            manifest = rows[0]['data']
            self.assertEqual(manifest['llm']['timeout'], 180)
            self.assertEqual(manifest['reasoning_effort'], 'none')
            self.assertNotIn('thinking', manifest)
            self.assertIn('observer_mean', manifest['perception_scope'])
            for entry in rows:
                if entry['kind'] == 'round':
                    nodes = entry['data']['nodes']
                    average = sum(n['after']['perceived_support'] for n in nodes if n['node'] in net.observers) / len(net.observers)
                    for node in nodes:
                        if node['node'] not in net.observers:
                            self.assertAlmostEqual(node['after']['perceived_support'], average)
                            self.assertEqual(node['perception']['model'], 'observer_mean')
            calls = [json.loads(line)['data'] for line in (output / 'calls.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertTrue(calls)
            self.assertTrue(all(c['context']['node'] in net.observers for c in calls))
            self.assertTrue(all(c['context']['reasoning_effort'] == 'none' for c in calls))
            self.assertTrue(any(json.loads(b['messages'][1]['content'])['history'] for b in requests))

    def test_sdk_configuration_and_actual_mock_http_body(self):
        with patch('majority.llm.OpenAI') as ctor:
            open_client(self.config, 'fake-test-key')
        self.assertEqual(ctor.call_args.kwargs['timeout'], 180)
        self.assertEqual(ctor.call_args.kwargs['max_retries'], 0)
        bodies = []
        def respond(request):
            self.assertEqual(str(request.url), 'http://127.0.0.1:3425/v1/chat/completions')
            body = json.loads(request.content)
            bodies.append(body)
            self.assertEqual(body['model'], 'qoder-cn/dfmodel')
            self.assertEqual(body['reasoning_effort'], 'none')
            self.assertEqual(body['max_tokens'], 4096)
            self.assertEqual(body['temperature'], 0)
            self.assertNotIn('thinking', body)
            self.assertNotIn('response_format', body)
            return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 1,
                'model': 'qoder-cn/dfmodel', 'choices': [{'index': 0, 'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': '{"support": 0.6}'}}]})
        for chat_type in (StudyChat, JsonChat):
            audit = io.StringIO()
            with OpenAI(api_key='fake-test-key', base_url=self.config.base_url, timeout=180, max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
                result, _ = chat_type(client, self.config, audit).request('perception', {}, lambda x:x['support'], temperature=0)
            self.assertEqual(result, .6)
            row = json.loads(audit.getvalue())['data']
            self.assertEqual(row['context']['reasoning_effort'], 'none')
            self.assertEqual(row['context']['base_url'], self.config.base_url)
            self.assertNotIn('thinking', row['context'])
        self.assertEqual(len(bodies), 2)

    def test_reasoning_change_does_not_reuse_cached_response(self):
        def respond(request):
            return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 1,
                'model': 'qoder-cn/dfmodel', 'choices': [{'index': 0, 'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': '{"support": 0.6}'}}]})
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'calls.jsonl'
            for effort, expected_reused in [('none', False), ('none', True), ('high', False)]:
                audit = io.StringIO()
                calls = []
                def handler(request):
                    calls.append(1)
                    return respond(request)
                with OpenAI(api_key='test', base_url=self.config.base_url, max_retries=0,
                            http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
                    chat = StudyChat(client, replace(self.config, reasoning_effort=effort), audit,
                                     path if path.exists() else None)
                    chat.request('perception', {}, lambda x:x['support'], temperature=0)
                row = json.loads(audit.getvalue())['data']
                self.assertEqual(row['reused'], expected_reused)
                self.assertEqual(len(calls), 0 if expected_reused else 1)
                if not path.exists():
                    path.write_text(audit.getvalue(), encoding='utf-8')

    def test_truncated_output_is_not_a_perception_estimate(self):
        from openai.types.chat import ChatCompletion
        response = ChatCompletion.model_validate({'id': 'truncated', 'object': 'chat.completion', 'created': 1,
            'model': 'qoder-cn/dfmodel', 'choices': [{'index': 0, 'finish_reason': 'length',
                'message': {'role': 'assistant', 'content': '{"support": 0.6}'}}]})
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:response)))
        with self.assertRaises(LLMValidationError):
            StudyChat(client, self.config, io.StringIO()).request('perception', {}, lambda x:x['support'], temperature=0)


if __name__ == '__main__':
    unittest.main()
