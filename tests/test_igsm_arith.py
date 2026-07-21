import importlib.util
import unittest
from pathlib import Path


UTILS_PATH = (
    Path(__file__).resolve().parents[1]
    / "lm_eval"
    / "tasks"
    / "igsm_arith"
    / "utils.py"
)
SPEC = importlib.util.spec_from_file_location("igsm_arith_utils", UTILS_PATH)
assert SPEC is not None and SPEC.loader is not None
igsm_arith_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(igsm_arith_utils)


class FakeDataset(list):
    def filter(self, predicate):
        return FakeDataset(row for row in self if predicate(row))


class IGSMArithmeticUtilsTest(unittest.TestCase):
    def setUp(self):
        self.dataset = FakeDataset(
            [
                {"id": "d2-a", "target_depth": 2},
                {"id": "d3-a", "target_depth": 3},
                {"id": "d4-a", "target_depth": 4},
                {"id": "d2-b", "target_depth": "2"},
            ]
        )

    def test_depth_filters(self):
        cases = [
            (igsm_arith_utils.process_docs_d2, ["d2-a", "d2-b"]),
            (igsm_arith_utils.process_docs_d3, ["d3-a"]),
            (igsm_arith_utils.process_docs_d4, ["d4-a"]),
        ]
        for process_docs, expected_ids in cases:
            with self.subTest(process_docs=process_docs.__name__):
                filtered = process_docs(self.dataset)
                self.assertEqual([row["id"] for row in filtered], expected_ids)

    def test_rejects_unsupported_depth(self):
        with self.assertRaisesRegex(ValueError, "Unsupported iGSM arithmetic depth"):
            igsm_arith_utils.process_docs_by_depth(self.dataset, 5)


if __name__ == "__main__":
    unittest.main()
