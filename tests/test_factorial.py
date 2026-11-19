"""Tests written for later approval; all API-facing behavior uses guards or mocks."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from majority.models import Network, Profile
from scripts.majority_factorial import (
    DEFAULT_PROTOCOL, PROJECT, audit_positions, build_tasks, digest, file_hash,
    inside, load_prepared, read_json, run_batch, run_directory, search_seed,
    seed_diagnostics, task_identity, task_settings, validate_protocol,
    validate_result, write_json,
)
from scripts.summarize_factorial import analyze_rows, collect_rows, holm, primary_contrasts, summarize_calls


class FactorialTests(unittest.TestCase):
    def setUp(self):
        self.protocol = read_json(DEFAULT_PROTOCOL)
        self.baseline = read_json(PROJECT / self.protocol["baseline"])
        self.groups = [{"graph": graph, "repeat": repeat,
                        "network_file": f"inputs/network-{graph}.json",
                        "profiles_file": f"inputs/g{graph}-r{repeat}-profiles.json",
                        "positions": {str(k): f"inputs/g{graph}-r{repeat}-k{k}.json"
                                      for k in self.protocol["k_grid"]}}
                       for graph in self.protocol["graphs"] for repeat in self.protocol["repeats"]]
        self.tasks = build_tasks(self.protocol, self.groups)

    def test_grid_common_positions_and_pilot_size(self):
        self.assertEqual(len(self.tasks), 108)
        self.assertEqual(sum(task["pilot"] for task in self.tasks), 9)
        for group in self.groups:
            for k in self.protocol["k_grid"]:
                tasks = [task for task in self.tasks
                         if (task["graph"], task["repeat"], task["k"]) == (group["graph"], group["repeat"], k)]
                self.assertEqual(len(tasks), 3)
                self.assertEqual(len({task["positions_file"] for task in tasks}), 1)
                self.assertEqual(len({task["group_seed"] for task in tasks}), 1)
                self.assertEqual({task["rho"] for task in tasks}, {1, 3, 6})

    def test_protocol_and_actual_condition_settings(self):
        validate_protocol(self.protocol, self.baseline)
        task = next(task for task in self.tasks if task["k"] == 100 and task["rho"] == 3)
        settings = task_settings(self.baseline, self.protocol, task)
        self.assertEqual(settings["fraction"], .1)
        self.assertEqual(settings["rho"], 3)
        self.assertEqual(settings["theta"], self.baseline["settings"]["theta"])
        for key, value in (("alpha", .2), ("beta", 5)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_protocol(self.protocol, {**self.baseline, key: value})

    def test_search_seed_and_condition_identity(self):
        seed = search_seed("main2-position-v1", 0, 1, 100)
        self.assertEqual(seed, search_seed("main2-position-v1", 0, 1, 100))
        self.assertNotEqual(seed, search_seed("main2-position-v1", 0, 1, 160))
        manifest = {"bundle_id": "a" * 64}
        first, second = self.tasks[:2]
        self.assertNotEqual(run_directory(Path("out"), manifest, first), run_directory(Path("out"), manifest, second))
        self.assertNotEqual(task_identity(manifest, first, {"model": "a"}),
                            task_identity(manifest, first, {"model": "b"}))
        self.assertNotEqual(task_identity(manifest, first, {}), task_identity(manifest, second, {}))

    def test_read_only_import_and_api_opt_in(self):
        with patch("scripts.majority_factorial.load_prepared") as loader:
            with self.assertRaises(ValueError):
                run_batch("does-not-exist", "pilot")
            loader.assert_not_called()

    def test_hand_computed_initial_exposure(self):
        network = Network((0,) * 6,
            tuple(frozenset(group) for group in ({1}, {0}, {0, 1}, {0}, {0, 1, 2}, {2, 3})),
            frozenset({4, 5}))
        profiles = tuple(Profile(g, .4) for g in (.2, .5, .2, .5, .2, .5))
        row = seed_diagnostics(network, profiles, {0, 1}, [1, 3])
        self.assertEqual(row["illusion"], .75)
        self.assertEqual(row["observer_illusion"], .5)
        self.assertEqual(row["fan_total"], 7)
        self.assertEqual(row["fan_union"], 5)
        self.assertAlmostEqual(row["exposure"][1]["expected_speakers"], 3 / 7 + 3 / 4)
        self.assertAlmostEqual(row["exposure"][1]["expected_deliveries"], 4 * 3 / 7 + 3 * 3 / 4)

    def test_position_audit_excludes_observers_and_keeps_zero_gap_warning(self):
        network = Network((0,) * 6,
                          tuple(frozenset(set(range(6)) - {i}) for i in range(6)), frozenset({4, 5}))
        profiles = (Profile(.2, .4),) * 6
        positions = {"TH": [0, 1], "TL": [0, 2], "U0": [1, 2]}
        audit = audit_positions(network, profiles, positions, 2, self.protocol["search"], [1, 3, 6])
        self.assertTrue(audit["warnings"])
        for gap in audit["matching"].values():
            self.assertTrue(all(value == 0 for value in gap.values()))
        with self.assertRaises(ValueError):
            audit_positions(network, profiles, {**positions, "TH": [0, 4]}, 2,
                            self.protocol["search"], [1, 3, 6])

    def test_frozen_input_edit_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "baseline.json", self.baseline)
            write_json(root / "sentinel.json", {"frozen": 1})
            manifest = {"protocol": self.protocol, "baseline_file": "baseline.json",
                        "groups": self.groups, "tasks": self.tasks, "source_hashes": {},
                        "files": {name: file_hash(root / name) for name in ("baseline.json", "sentinel.json")}}
            manifest["bundle_id"] = digest(manifest)
            write_json(root / "manifest.json", manifest)
            self.assertEqual(load_prepared(root, check_code=False)["tasks"], self.tasks)
            (root / "sentinel.json").write_text('{"frozen": 2}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_prepared(root, check_code=False)
            with self.assertRaises(ValueError):
                inside(root, "../outside.json")

    def result_fixture(self):
        task = next(task for task in self.tasks if task["k"] == 200 and task["rho"] == 3)
        settings = task_settings(self.baseline, self.protocol, task)
        summary = {"perceived_majority": .4, "feed_majority": .5,
                   "advocate_perception_bias": .3, "public_conformity": .1, "opponent_silence": .7}
        result = {"branch": "TH", "size": 1000, "k": 200, "seed": task["group_seed"],
                  "settings": settings, "tail_rounds": 3, "summary": summary,
                  "metrics": [{"round": i, "true_advocate_support": .2, **summary} for i in range(1, 11)]}
        return task, settings, result

    def test_wrong_condition_partial_rounds_and_summary_cannot_be_reused(self):
        task, settings, result = self.result_fixture()
        validate_result(result, task, self.baseline, settings)
        altered = []
        wrong_rho = deepcopy(result)
        wrong_rho["settings"]["rho"] = 6
        altered.append(wrong_rho)
        altered.append({**result, "metrics": result["metrics"][:-1]})
        wrong_truth = deepcopy(result)
        wrong_truth["metrics"][0]["true_advocate_support"] = .3
        altered.append(wrong_truth)
        wrong_tail = deepcopy(result)
        wrong_tail["summary"]["perceived_majority"] = .9
        altered.append(wrong_tail)
        for value in altered:
            with self.assertRaises(ValueError):
                validate_result(value, task, self.baseline, settings)

    def test_missing_results_remain_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            write_json(Path(directory) / "runtime.json", {})
            manifest = {"tasks": self.tasks, "bundle_id": "b" * 64}
            rows, missing = collect_rows(directory, manifest, pilot=True)
            self.assertEqual(rows, [])
            self.assertEqual(len(missing), 9)

    def test_usage_excludes_replay_and_retains_unknown_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calls.jsonl"
            entries = [
                {"status": "ok", "reused": False, "response": {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}},
                {"status": "ok", "reused": True, "response": {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}},
                {"status": "api_error", "reused": False},
            ]
            path.write_text("\n".join(json.dumps({"kind": "llm_call", "data": row}) for row in entries), encoding="utf-8")
            counts = summarize_calls(path)
            self.assertEqual(counts["actual_calls"], 2)
            self.assertEqual(counts["reused_rows"], 1)
            self.assertEqual(counts["known_input_tokens"], 10)
            self.assertEqual(counts["calls_without_complete_usage"], 1)


class FactorialAnalysisTests(unittest.TestCase):
    def fixture(self):
        protocol = read_json(DEFAULT_PROTOCOL)
        protocol["k_grid"] = [100, 200]
        protocol["rho_grid"] = [1, 6]
        protocol["analysis"]["bootstrap_repeats"] = 100
        rows = [{"graph": graph, "repeat": repeat, "k": k, "rho": rho,
                 "q_tail": .1 + .05 * x + .1 * y + .04 * x * y}
                for graph in range(3) for repeat in range(3)
                for x, k in enumerate((100, 200)) for y, rho in enumerate((1, 6))]
        return protocol, rows

    def test_hand_computed_contrasts_and_exact_sign_flip(self):
        protocol, rows = self.fixture()
        values = {(row["k"], row["rho"]): row["q_tail"] for row in rows}
        contrast = primary_contrasts(values, protocol["k_grid"], protocol["rho_grid"])
        for actual, expected in zip(contrast, (.07, .12, .04)):
            self.assertAlmostEqual(actual, expected)
        result = analyze_rows(rows, protocol)
        self.assertEqual(result["paired_groups"], 9)
        for row, expected in zip(result["comparisons"], (.07, .12, .04)):
            self.assertAlmostEqual(row["difference"], expected)
            self.assertAlmostEqual(row["p_value"], 2 / 512)
            self.assertAlmostEqual(row["p_holm"], 6 / 512)
        self.assertTrue(all(value is None for value in result["smallest_reached_k_in_tested_grid"].values()))

    def test_duplicate_and_incomplete_grid_are_rejected(self):
        protocol, rows = self.fixture()
        for incomplete in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                analyze_rows(incomplete, protocol)
        self.assertEqual(holm([.01, .04, .2]), [.03, .08, .2])


if __name__ == "__main__":
    unittest.main()
