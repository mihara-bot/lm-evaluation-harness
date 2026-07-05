import ast
import re
from typing import Tuple

import datasets


CHOICES = [
    "aunt",
    "son-in-law",
    "grandfather",
    "brother",
    "sister",
    "father",
    "mother",
    "grandmother",
    "uncle",
    "daughter-in-law",
    "grandson",
    "granddaughter",
    "father-in-law",
    "mother-in-law",
    "nephew",
    "son",
    "daughter",
    "niece",
]
RELATION_TO_LABEL = {relation: idx for idx, relation in enumerate(CHOICES)}


def _normalize_relation(relation: str) -> str:
    return str(relation).strip().lower()


def _parse_query(query) -> Tuple[str, str]:
    if isinstance(query, (list, tuple)) and len(query) == 2:
        return str(query[0]), str(query[1])

    query_text = str(query).strip()
    try:
        parsed = ast.literal_eval(query_text)
    except (SyntaxError, ValueError):
        parsed = None

    if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
        return str(parsed[0]), str(parsed[1])

    names = re.findall(r"[A-Za-z][A-Za-z'-]*", query_text)
    if len(names) >= 2:
        return names[0], names[1]

    raise ValueError(f"Could not parse CLUTRR query: {query!r}")


def _process_doc(doc: dict) -> dict:
    relation = _normalize_relation(doc["target_text"])
    if relation not in RELATION_TO_LABEL:
        raise ValueError(f"Unexpected CLUTRR target_text label: {doc['target_text']!r}")

    query_person1, query_person2 = _parse_query(doc["query"])

    return {
        "id": doc["id"],
        "story": doc["story"],
        "clean_story": doc.get("clean_story"),
        "query": doc["query"],
        "query_person1": query_person1,
        "query_person2": query_person2,
        "target": doc.get("target"),
        "target_text": relation,
        "label": RELATION_TO_LABEL[relation],
        "choices": CHOICES,
        "task_name": doc.get("task_name"),
        "proof_state": doc.get("proof_state"),
        "f_comb": doc.get("f_comb"),
        "story_edges": doc.get("story_edges"),
        "edge_types": doc.get("edge_types"),
        "query_edge": doc.get("query_edge"),
        "genders": doc.get("genders"),
        "task_split": doc.get("task_split"),
    }


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.map(_process_doc)


def process_docs_by_task_name(
    dataset: datasets.Dataset, task_name: str
) -> datasets.Dataset:
    return dataset.filter(lambda doc: doc.get("task_name") == task_name).map(
        _process_doc
    )


def process_docs_task_1_2(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs_by_task_name(dataset, "task_1.2")


def process_docs_task_1_3(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs_by_task_name(dataset, "task_1.3")


def process_docs_task_1_4(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs_by_task_name(dataset, "task_1.4")


def process_docs_task_1_5(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs_by_task_name(dataset, "task_1.5")


def process_docs_task_1_6(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs_by_task_name(dataset, "task_1.6")


def process_docs_task_1_7(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs_by_task_name(dataset, "task_1.7")


def process_docs_task_1_8(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs_by_task_name(dataset, "task_1.8")


def process_docs_task_1_9(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs_by_task_name(dataset, "task_1.9")


def process_docs_task_1_10(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs_by_task_name(dataset, "task_1.10")
