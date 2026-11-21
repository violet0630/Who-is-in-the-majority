import json
from pathlib import Path
import unittest

from scripts.ext_resistance import CONDITIONS, BASE_SETTINGS, settings_for


class ExpressionProtocolTests(unittest.TestCase):
    def test_conditions_are_expression_parameters(self):
        self.assertEqual(set(CONDITIONS), {"E0", "E1", "E2", "E3"})
        self.assertEqual(CONDITIONS["E1"]["theta_d"], -2.0)
        self.assertEqual(CONDITIONS["E2"]["theta_d"], 0.0)
        self.assertEqual(CONDITIONS["E3"]["kappa"], 1.0)
        self.assertNotIn("private_update", CONDITIONS["E0"])

    def test_v2_settings_are_fifteen_rounds(self):
        for condition in CONDITIONS:
            settings = settings_for(condition)
            self.assertEqual(settings["rounds"], 15)
            self.assertEqual(settings["tail_rounds"], 5)
            self.assertEqual(settings["read_limit"], 11)
            self.assertEqual(settings["max_comments"], 3)
            self.assertEqual(settings["comment_alpha"], 1.0)
            self.assertEqual(settings["comment_beta"], 9.0)

    def test_conditions_change_only_expression_fields(self):
        base = settings_for("E0")
        self.assertEqual({k for k in settings_for("E1") if settings_for("E1")[k] != base[k]}, {"theta"})
        self.assertEqual({k for k in settings_for("E2") if settings_for("E2")[k] != base[k]}, {"theta"})
        self.assertEqual({k for k in settings_for("E3") if settings_for("E3")[k] != base[k]}, {"kappa"})


if __name__ == "__main__":
    unittest.main()
