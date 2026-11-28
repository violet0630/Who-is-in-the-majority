import io
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from openai.types.chat import ChatCompletion

from majority.llm import LLMConfig, prompt_hash
from majority.nonthinking import NonThinkingError, NonThinkingGuard
from scripts.majority_study import StudyChat, study_runtime


def response(*, reasoning_tokens=0, output=30, usage=True, content='{"support":0.5}', **fields):
    raw = {'id': 'offline-response', 'object': 'chat.completion', 'created': 0,
           'model': 'qoder-cn/dfmodel', 'choices': [{'index': 0, 'finish_reason': 'stop',
               'message': {'role': 'assistant', 'content': content, **fields}}]}
    if usage:
        raw['usage'] = {'prompt_tokens': 10, 'completion_tokens': output, 'total_tokens': output+10,
                        'completion_tokens_details': {'reasoning_tokens': reasoning_tokens}}
    return ChatCompletion.model_validate(raw)


class NonthinkingRuntimeTests(unittest.TestCase):
    config = LLMConfig(base_url='http://127.0.0.1:3425/v1', model='qoder-cn/dfmodel',
                       timeout=180, max_tokens=4096, reasoning_effort='none')

    def chat(self, guard, value=None, reuse=None):
        method = Mock(return_value=value or response())
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=method)))
        audit = io.StringIO()
        chat = StudyChat(client, self.config, audit, reuse, nonthinking_guard=guard)
        return chat, method, audit

    def test_visible_anomalies_rejected_and_not_retried(self):
        values = [response(reasoning_content='reasoning'), response(reasoning_tokens=1),
                  response(output=4097), response(reasoning='reasoning'),
                  response(reasoning_details=[{'text': 'reasoning'}]),
                  response(content='```xml\n<THINK>reasoning</THINK>')]
        for value in values:
            with self.subTest(value=value.id):
                guard = NonThinkingGuard()
                chat, method, audit = self.chat(guard, value)
                with self.assertRaises(NonThinkingError):
                    chat.request('perception', {}, lambda d:d['support'], temperature=0)
                self.assertEqual(method.call_count, 1)
                row = json.loads(audit.getvalue())['data']
                self.assertEqual(row['status'], 'rejected_nonthinking')
                self.assertEqual(row['response']['id'], value.id)
                second, second_method, _ = self.chat(guard)
                with self.assertRaises(NonThinkingError):
                    second.request('perception', {}, lambda d:d['support'], temperature=0)
                second_method.assert_not_called()

    def test_missing_usage_is_recorded_unknown_not_zero(self):
        chat, method, audit = self.chat(NonThinkingGuard(), response(usage=False))
        value, _ = chat.request('perception', {}, lambda d:d['support'], temperature=0)
        self.assertEqual(value, .5)
        check = json.loads(audit.getvalue())['data']['nonthinking_check']
        self.assertFalse(check['usage_reported'])
        self.assertIsNone(check['reasoning_tokens'])

    def test_output_limit_is_inclusive_and_stop_survives_restart(self):
        with TemporaryDirectory() as folder:
            stop = Path(folder) / 'nonthinking_stop.json'
            guard = NonThinkingGuard(stop)
            guard.inspect(response(output=4096), 4096)
            with self.assertRaises(NonThinkingError):
                guard.inspect(response(reasoning_tokens=7), 4096)
            self.assertEqual(json.loads(stop.read_text())['reason'], 'positive_reasoning_tokens')
            with self.assertRaises(NonThinkingError):
                NonThinkingGuard(stop).check()

    def test_waiting_on_semaphore_checks_stop_again_before_dispatch(self):
        guard = NonThinkingGuard()
        class StopWhileWaiting:
            def __enter__(self):
                try:
                    guard.inspect(response(reasoning_tokens=1), 4096)
                except NonThinkingError:
                    pass
            def __exit__(self, *args):
                return False
        chat, method, _ = self.chat(guard)
        with patch('scripts.majority_study.GATEWAY_SLOTS', StopWhileWaiting()):
            with self.assertRaises(NonThinkingError):
                chat.request('perception', {}, lambda d:d['support'], temperature=0)
        method.assert_not_called()

    def test_reused_response_is_checked_before_accepting(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'calls.jsonl'
            context = {'reasoning_effort': 'none', 'base_url': self.config.base_url}
            path.write_text(json.dumps({'data': {'status': 'ok', 'task': 'perception',
                'payload': {}, 'context': context, 'prompt_sha256': prompt_hash('perception'),
                'model': self.config.model, 'temperature': 0, 'max_tokens': 4096,
                'response': response(reasoning_tokens=2).model_dump(mode='json')}})+'\n',encoding='utf-8')
            chat, method, audit = self.chat(NonThinkingGuard(), reuse=path)
            with self.assertRaises(NonThinkingError):
                chat.request('perception', {}, lambda d:d['support'], temperature=0)
            method.assert_not_called()
            row = json.loads(audit.getvalue())['data']
            self.assertTrue(row['reused'])
            self.assertEqual(row['status'], 'rejected_nonthinking')

    def test_batch_stops_other_tasks_and_refuses_automatic_restart(self):
        from scripts.majority_factorial import run_batch
        manifest = {'protocol': {'llm': asdict(self.config), 'execution': {'run_workers': 1}},
                    'tasks': [{'id': str(i), 'pilot': True} for i in range(3)]}
        guards = []
        def execute(*args, nonthinking_guard=None):
            guards.append(nonthinking_guard)
            nonthinking_guard.inspect(response(reasoning_content='not allowed'), 4096)
        with TemporaryDirectory() as folder:
            with patch('scripts.majority_factorial.load_prepared', return_value=manifest), \
                 patch('majority.llm.load_settings', return_value=(self.config, 'test-key')), \
                 patch('scripts.majority_factorial.execute_task', side_effect=execute) as task:
                with self.assertRaises(RuntimeError):
                    run_batch(Path(folder), 'pilot', allow_api=True)
                self.assertEqual(task.call_count, 1)
                self.assertTrue((Path(folder)/'nonthinking_stop.json').exists())
                with self.assertRaises(NonThinkingError):
                    run_batch(Path(folder), 'pilot', allow_api=True)
                self.assertEqual(task.call_count, 1)
        self.assertEqual(study_runtime(self.config)['nonthinking_guard_version'], 1)


if __name__ == '__main__':
    unittest.main()
