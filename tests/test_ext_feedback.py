import unittest

from scripts.ext_feedback import CONDITIONS, derive_settings


class FeedbackProtocolTests(unittest.TestCase):
    def test_conditions_cover_beta_and_alpha(self):
        self.assertEqual(set(CONDITIONS), {"FB0", "FB1", "FB2", "FB3", "FB4"})
        self.assertEqual(CONDITIONS["FB1"], {"beta": 0})
        self.assertEqual(CONDITIONS["FB2"], {"beta": 50})
        self.assertEqual(CONDITIONS["FB3"], {"alpha": 0.1})
        self.assertEqual(CONDITIONS["FB4"], {"alpha": 0.5})

    def test_derive_settings_changes_one_field_only(self):
        base = {"alpha": 0.05, "beta": 10, "theta": [1, 0, -1, 0]}
        for name, override in CONDITIONS.items():
            settings = derive_settings(base, name)
            changes = {k for k in settings if settings[k] != base.get(k)}
            self.assertEqual(changes, set(override) or set(), name)
            self.assertEqual({k: settings[k] for k in override}, override)


if __name__ == "__main__":
    unittest.main()
