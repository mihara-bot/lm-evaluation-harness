import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


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
DATASETS_STUB = types.ModuleType("datasets")
DATASETS_STUB.Dataset = object
DATASETS_STUB.DownloadManager = object
with mock.patch.dict(sys.modules, {"datasets": DATASETS_STUB}):
    SPEC.loader.exec_module(ruletaker_utils)


class RuleTakerUtilsTest(unittest.TestCase):
    def test_find_data_file_supports_extra_archive_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            expected = (
                root
                / "extra-cache-level"
                / "rule-reasoning-dataset-V2020.2.5.0"
                / "original"
                / "depth-0"
                / "meta-test.jsonl"
            )
            expected.parent.mkdir(parents=True)
            expected.touch()

            # A similarly named file from another archive variant must not win.
            problog = root / "problog" / "depth-0" / "meta-test.jsonl"
            problog.parent.mkdir(parents=True)
            problog.touch()

            self.assertEqual(ruletaker_utils._find_data_file(root, 0), expected)

    def test_find_data_file_accepts_direct_file_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            expected = Path(temp_dir) / "depth-3" / "meta-test.jsonl"
            expected.parent.mkdir(parents=True)
            expected.touch()
            self.assertEqual(ruletaker_utils._find_data_file(expected, 3), expected)

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
