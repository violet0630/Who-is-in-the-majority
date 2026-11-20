import unittest

from majority.models import RunConfig
from scripts.majority_study import coefficients_from


class ExtensionSettingsTests(unittest.TestCase):
    def test_coefficients_beta_from_settings(self):
        base = {"theta": [1, 0, -1, 0], "kappa": 0.5, "gamma": 1}
        self.assertEqual(coefficients_from(base).beta, 10.0)          # 缺省回退
        self.assertEqual(coefficients_from({**base, "beta": 50}).beta, 50.0)
        self.assertEqual(coefficients_from({**base, "beta": 0}).beta, 0.0)

    def test_alpha_default_unchanged(self):
        self.assertEqual(RunConfig(rounds=10, alpha=0.05).alpha, 0.05)
        self.assertEqual(RunConfig(rounds=10).alpha, 0.05)


if __name__ == "__main__":
    unittest.main()
