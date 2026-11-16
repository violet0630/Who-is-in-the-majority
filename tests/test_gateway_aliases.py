import io
import json
from pathlib import Path
import tempfile
import unittest

import httpx
from openai import OpenAI

from majority.llm import JsonChat, LLMConfig, canonical_model


class GatewayAliasTests(unittest.TestCase):
    def test_model_name_follows_gateway(self):
        remote = LLMConfig(base_url="https://api.1nvictus.me/v1", model="deepseek-v4-flash")
        local = LLMConfig(base_url="http://127.0.0.1:3000/v1", model="deepseek-flash")
        self.assertEqual(remote.model, "deepseek-flash")
        self.assertEqual(local.model, "deepseek-v4-flash")
        self.assertEqual(canonical_model(remote.model), canonical_model(local.model))
        self.assertEqual(LLMConfig(model="another-model").model, "another-model")

    def test_cached_response_survives_gateway_switch_both_directions(self):
        remote = LLMConfig(base_url="https://api.1nvictus.me/v1", model="deepseek-flash")
        local = LLMConfig(base_url="http://127.0.0.1:3000/v1", model="deepseek-v4-flash")
        for first, second in ((remote, local), (local, remote)):
            with self.subTest(first=first.base_url), tempfile.TemporaryDirectory() as directory:
                calls = []

                def respond(request):
                    body = json.loads(request.content)
                    calls.append(body["model"])
                    self.assertEqual(body["model"], first.model)
                    return httpx.Response(200, json={
                        "id": "saved-response", "object": "chat.completion", "created": 1,
                        "model": "deepseek-flash", "choices": [{"index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant", "content": '{"support": 0.6}'}}],
                    })

                path = Path(directory) / "calls.jsonl"
                with OpenAI(api_key="test-key", base_url=first.base_url, max_retries=0,
                            http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
                    with path.open("x", encoding="utf-8") as audit:
                        value, _ = JsonChat(client, first, audit).request("perception", {"sample": 1},
                            lambda data: data["support"], temperature=0, context={"trial": 1})
                self.assertEqual(value, 0.6)

                def unexpected(request):
                    raise AssertionError("A compatible cached response must avoid a new API call")

                audit = io.StringIO()
                with OpenAI(api_key="test-key", base_url=second.base_url, max_retries=0,
                            http_client=httpx.Client(transport=httpx.MockTransport(unexpected))) as client:
                    value, completion = JsonChat(client, second, audit, path).request("perception", {"sample": 1},
                        lambda data: data["support"], temperature=0, context={"trial": 1})
                self.assertEqual(value, 0.6)
                self.assertEqual(completion.request_id, "saved-response")
                self.assertEqual(calls, [first.model])
                record = json.loads(audit.getvalue())["data"]
                self.assertTrue(record["reused"])
                self.assertEqual(record["model"], second.model)
                self.assertEqual(record["canonical_model"], "deepseek-flash")
                self.assertNotIn("test-key", audit.getvalue())


if __name__ == "__main__":
    unittest.main()
