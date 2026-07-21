import importlib.util
import unittest
from pathlib import Path


UTILS_PATH = (
    Path(__file__).resolve().parents[1]
    / "lm_eval"
    / "tasks"
    / "ruletaker"
    / "utils.py"
)
SPEC = importlib.util.spec_from_file_location("ruletaker_utils", UTILS_PATH)
assert SPEC is not None and SPEC.loader is not None
ruletaker_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ruletaker_utils)


class RuleTakerUtilsTest(unittest.TestCase):
    def test_build_theory_uses_natural_triple_then_rule_order(self):
        record = {
            "triples": {
                "triple10": {"text": "Charlie is blue."},
                "triple2": {"text": "Bob is red."},
                "triple1": {"text": "Alice is green."},
            },
            "rules": {
                "rule2": {"text": "Red things are quiet."},
                "rule1": {"text": "Green things are kind."},
            },
        }
        self.assertEqual(
            ruletaker_utils._build_theory(record),
            "Alice is green. Bob is red. Charlie is blue. "
            "Green things are kind. Red things are quiet.",
        )

    def test_flatten_records_normalizes_labels_and_preserves_question_order(self):
        records = [
            {
                "theory": "Bob is red.",
                "questions": {
                    "Q2": {"question": "Bob is cold.", "answer": "false"},
                    "Q1": {
                        "id": "q1",
                        "question": "Bob is red.",
                        "answer": True,
                    },
                },
            }
        ]
        self.assertEqual(
            ruletaker_utils._flatten_records(records, depth=2),
            [
                {
                    "id": "q1",
                    "theory": "Bob is red.",
                    "question": "Bob is red.",
                    "label": 0,
                    "depth": 2,
                },
                {
                    "id": "depth-2:0:Q2",
                    "theory": "Bob is red.",
                    "question": "Bob is cold.",
                    "label": 1,
                    "depth": 2,
                },
            ],
        )

    def test_normalize_depth(self):
        cases = [(0, 0), ("d1", 1), ("depth-2", 2), ("depth3", 3), ("5", 5)]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(ruletaker_utils._normalize_depth(value), expected)

    def test_normalize_depth_rejects_unsupported_depth(self):
        with self.assertRaisesRegex(ValueError, "Unsupported RuleTaker depth"):
            ruletaker_utils._normalize_depth(4)


if __name__ == "__main__":
    unittest.main()
