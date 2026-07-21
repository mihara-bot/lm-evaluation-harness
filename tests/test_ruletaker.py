import importlib.util
from pathlib import Path

import pytest


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


def test_build_theory_uses_natural_triple_then_rule_order():
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
    assert ruletaker_utils._build_theory(record) == (
        "Alice is green. Bob is red. Charlie is blue. "
        "Green things are kind. Red things are quiet."
    )


def test_flatten_records_normalizes_labels_and_preserves_question_order():
    records = [
        {
            "theory": "Bob is red.",
            "questions": {
                "Q2": {"question": "Bob is cold.", "answer": "false"},
                "Q1": {"id": "q1", "question": "Bob is red.", "answer": True},
            },
        }
    ]
    assert ruletaker_utils._flatten_records(records, depth=2) == [
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
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, 0), ("d1", 1), ("depth-2", 2), ("depth3", 3), ("5", 5)],
)
def test_normalize_depth(value, expected):
    assert ruletaker_utils._normalize_depth(value) == expected


def test_normalize_depth_rejects_unsupported_depth():
    with pytest.raises(ValueError, match="Unsupported RuleTaker depth"):
        ruletaker_utils._normalize_depth(4)
