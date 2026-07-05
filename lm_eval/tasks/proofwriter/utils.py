import re
from typing import Optional

import datasets


CHOICES = ["True", "False", "Unknown"]
ANSWER_TO_LABEL = {answer.lower(): idx for idx, answer in enumerate(CHOICES)}
DEPTH_PATTERN = re.compile(r"(?:depth|maxd|d)[_-]?(\d+)", re.IGNORECASE)


def _normalize_answer(answer: str) -> str:
    return str(answer).strip().lower()


def _doc_depth(doc: dict) -> Optional[int]:
    question_depth = doc.get("QDep")
    if question_depth is not None:
        try:
            return int(question_depth)
        except (TypeError, ValueError):
            pass

    max_depth = doc.get("maxD")
    if max_depth is not None:
        try:
            return int(max_depth)
        except (TypeError, ValueError):
            pass

    config = str(doc.get("config", ""))
    match = DEPTH_PATTERN.search(config)
    if match:
        return int(match.group(1))

    return None


def _process_doc(doc: dict) -> dict:
    answer = _normalize_answer(doc["answer"])
    if answer not in ANSWER_TO_LABEL:
        raise ValueError(f"Unexpected ProofWriter answer label: {doc['answer']!r}")

    return {
        "id": doc["id"],
        "theory": doc["theory"],
        "question": doc["question"],
        "answer": doc["answer"],
        "label": ANSWER_TO_LABEL[answer],
        "choices": CHOICES,
        "maxD": doc.get("maxD"),
        "QDep": doc.get("QDep"),
        "config": doc.get("config"),
    }


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.map(_process_doc)


def process_docs_by_depth(dataset: datasets.Dataset, depth: int) -> datasets.Dataset:
    return dataset.filter(lambda doc: _doc_depth(doc) == depth).map(_process_doc)


def process_docs_depth0(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs_by_depth(dataset, 0)


def process_docs_depth1(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs_by_depth(dataset, 1)


def process_docs_depth2(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs_by_depth(dataset, 2)


def process_docs_depth3(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs_by_depth(dataset, 3)


def process_docs_depth4(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs_by_depth(dataset, 4)


def process_docs_depth5(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs_by_depth(dataset, 5)
