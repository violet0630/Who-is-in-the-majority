from copy import deepcopy
import unittest
from unittest.mock import patch

from scripts.resume_main2_gateway import continue_runs, validate_transport_switch


class GatewayContinuationTests(unittest.TestCase):
    def setUp(self):
        self.original = {"llm": {"base_url": "https://api.deepseek.com", "model": "deepseek-flash",
                                  "timeout": 180, "max_tokens": 4096},
                         "temperature": 0, "thinking": "disabled", "packages": {"openai": "test"}}
        self.url = "https://api.1nvictus.me/v1"
        self.current = deepcopy(self.original)
        self.current["llm"]["base_url"] = self.url

    def test_only_explicit_route_change_allowed(self):
        validate_transport_switch(self.original, self.current, self.url)
        with self.assertRaises(ValueError):
            validate_transport_switch(self.original, self.current, "https://unapproved.invalid")
        for field, value in (("model", "another-model"), ("max_tokens", 2048), ("timeout", 90)):
            changed = deepcopy(self.current)
            changed["llm"][field] = value
            with self.assertRaises(ValueError):
                validate_transport_switch(self.original, changed, self.url)
        for field, value in (("temperature", .7), ("thinking", "enabled"), ("packages", {})):
            changed = {**self.current, field: value}
            with self.assertRaises(ValueError):
                validate_transport_switch(self.original, changed, self.url)

    def test_execution_requires_explicit_opt_in(self):
        with patch("scripts.resume_main2_gateway.load_prepared") as loader:
            with self.assertRaises(ValueError):
                continue_runs("missing", self.url)
            loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
