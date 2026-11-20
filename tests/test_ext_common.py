import unittest

from majority.models import RunConfig
from scripts.ext_common import holm, paired_condition_inference, unpaired_condition_inference
from scripts.majority_study import coefficients_from


def rows(condition, value):
    return [{"condition": condition, "graph": g, "repeat": r,
             "summary": {"perceived_majority": value},
             "metrics": [{"perceived_majority": value} for _ in range(10)]}
            for g in range(3) for r in range(3)]


class ExtensionSettingsTests(unittest.TestCase):
    def test_coefficients_beta_from_settings(self):
        base = {"theta": [1, 0, -1, 0], "kappa": 0.5, "gamma": 1}
        self.assertEqual(coefficients_from(base).beta, 10.0)          # 缺省回退
        self.assertEqual(coefficients_from({**base, "beta": 50}).beta, 50.0)
        self.assertEqual(coefficients_from({**base, "beta": 0}).beta, 0.0)

    def test_alpha_default_unchanged(self):
        self.assertEqual(RunConfig(rounds=10, alpha=0.05).alpha, 0.05)
        self.assertEqual(RunConfig(rounds=10).alpha, 0.05)


class InferenceTests(unittest.TestCase):
    def test_paired_inference_matches_sign_flip(self):
        data = rows("A", 0.4) + rows("B", 0.1)
        out = paired_condition_inference(data, [("A", "B")])
        self.assertAlmostEqual(out["comparisons"][0]["difference"], 0.3)
        self.assertAlmostEqual(out["comparisons"][0]["p_value"], 2 / 512)
        self.assertAlmostEqual(out["comparisons"][0]["p_holm"], 2 / 512)
        self.assertTrue(out["paired"])

    def test_paired_inference_rejects_unmatched_group(self):
        data = rows("A", 0.4) + rows("B", 0.1)
        data[0] = {**data[0], "graph": 9}
        with self.assertRaises(ValueError):
            paired_condition_inference(data, [("A", "B")])

    def test_unpaired_inference_difference(self):
        data = rows("A", 0.5) + rows("B", 0.1)
        out = unpaired_condition_inference(data, [("A", "B")])
        self.assertAlmostEqual(out["comparisons"][0]["difference"], 0.4, places=6)
        self.assertFalse(out["paired"])

    def test_unpaired_inference_requires_three_graphs(self):
        data = rows("A", 0.5) + rows("B", 0.1)
        data[0] = {**data[0], "graph": 9}
        with self.assertRaises(AssertionError):
            unpaired_condition_inference(data, [("A", "B")])

    def test_holm_order(self):
        self.assertEqual(holm([0.01, 0.03, 0.9]), [0.03, 0.06, 0.9])
        self.assertEqual(holm([0.01]), [0.01])


if __name__ == "__main__":
    unittest.main()
