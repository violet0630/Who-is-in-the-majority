import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import httpx
from openai import APIConnectionError
from openai.types.chat import ChatCompletion
from scripts.gateway_nonthinking import Circuit, GatewayGuardError, GuardedClient, quarantine_cache


def response(reason="", reasoning_tokens=None, tokens=30):
    usage = {"prompt_tokens": 10, "completion_tokens": tokens, "total_tokens": 10+tokens}
    if reasoning_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return ChatCompletion.model_validate({"id": "test-id", "object": "chat.completion", "created": 0,
        "model": "dfmodel", "choices": [{"index": 0, "finish_reason": "stop",
        "message": {"role": "assistant", "content": '{"support":0.5}', "reasoning_content": reason}}], "usage": usage})


class NonthinkingGatewayTests(unittest.TestCase):
    kwargs = {"model": "deepseek-flash", "messages": [{"role": "user", "content": "test"}],
              "temperature": 0, "max_tokens": 4096, "extra_body": {"thinking": {"type": "disabled"}}}

    def client(self, directory, value):
        method = Mock(return_value=value)
        base = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=method)), close=Mock())
        return GuardedClient(base, directory, Circuit()), method

    def test_injects_none_and_logs_actual_response(self):
        with TemporaryDirectory() as directory:
            client, method = self.client(directory, response())
            try:
                client.create(**self.kwargs)
                self.assertEqual(method.call_args.kwargs['reasoning_effort'], 'none')
                self.assertEqual(method.call_args.kwargs['max_tokens'], 4096)
                row=json.loads((Path(directory)/'nonthinking_transport.jsonl').read_text(encoding='utf-8'))
                self.assertEqual(row['status'],'accepted_transport')
                self.assertNotIn('api_key',row)
            finally:client.close()

    def test_anomalies_trip_circuit_and_block_next_request(self):
        for value in (response(reason='internal reasoning'), response(reasoning_tokens=1), response(tokens=4097)):
            with self.subTest(value=value), TemporaryDirectory() as directory:
                client, method = self.client(directory, value)
                try:
                    with self.assertRaises(GatewayGuardError):client.create(**self.kwargs)
                    with self.assertRaises(GatewayGuardError):client.create(**self.kwargs)
                    self.assertEqual(method.call_count,1)
                finally:client.close()

    def test_three_transport_errors_stop_dispatch(self):
        with TemporaryDirectory() as directory:
            client, method = self.client(directory, response())
            method.side_effect=APIConnectionError(request=httpx.Request('POST','https://example.invalid'))
            try:
                for _ in range(3):
                    with self.assertRaises(APIConnectionError):client.create(**self.kwargs)
                with self.assertRaises(GatewayGuardError):client.create(**self.kwargs)
                self.assertEqual(method.call_count,3)
            finally:client.close()

    def test_cache_quarantine_preserves_trusted_prefix_and_is_idempotent(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            prefix=b'{"status":"trusted"}\n'
            full=prefix+b'{"status":"uncertain"}\n'
            (root/'calls.jsonl').write_bytes(full)
            (root/'gateway-continuation-001.json').write_text(json.dumps({
                'previous_calls_bytes':len(prefix),'previous_calls_sha256':hashlib.sha256(prefix).hexdigest()}),encoding='utf-8')
            marker=quarantine_cache(root)
            self.assertEqual((root/'calls.jsonl').read_bytes(),prefix)
            self.assertEqual((root/marker['backup_file']).read_bytes(),full)
            new=prefix+b'{"status":"guarded_new"}\n'
            (root/'calls.jsonl').write_bytes(new)
            self.assertEqual(quarantine_cache(root),marker)
            self.assertEqual((root/'calls.jsonl').read_bytes(),new)
            (root/'calls.jsonl').write_bytes(b'changed')
            with self.assertRaises(ValueError):quarantine_cache(root)

    def test_direct_only_cache_is_preserved_and_completed_runs_refused(self):
        with TemporaryDirectory() as directory:
            root=Path(directory); data=b'{"status":"trusted"}\n'
            (root/'calls.jsonl').write_bytes(data)
            marker=quarantine_cache(root)
            self.assertEqual(marker['quarantined_bytes'],0)
            self.assertEqual((root/'calls.jsonl').read_bytes(),data)
            (root/'complete.json').write_text('{}')
            with self.assertRaises(ValueError):quarantine_cache(root)

    def test_bad_boundary_refused_before_mutation(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);data=b'{"status":"trusted"}\n'
            (root/'calls.jsonl').write_bytes(data)
            (root/'gateway-continuation-001.json').write_text(json.dumps({
                'previous_calls_bytes':len(data),'previous_calls_sha256':'invalid'}),encoding='utf-8')
            with self.assertRaises(ValueError):quarantine_cache(root)
            self.assertEqual((root/'calls.jsonl').read_bytes(),data)


if __name__ == '__main__':unittest.main()
