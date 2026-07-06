import json
import re
from typing import Iterable
from urllib.request import urlopen

import datasets


BASE_URL = "https://huggingface.co/datasets/tasksource/prontoqa/raw/main"
HOP_FILES = {
    1: "1hop_ProofsOnly_random_noadj.json",
    2: "2hop_ProofsOnly_random_noadj.json",
    3: "3hop_ProofsOnly_random_noadj.json",
    4: "4hop_ProofsOnly_random_noadj.json",
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip()).lower()


def _load_json(url: str) -> dict:
    with urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _format_chain(chain: Iterable[str]) -> str:
    return " ".join(str(step).strip() for step in chain if str(step).strip())


def _negate_statement(statement: str) -> str:
    statement = str(statement).strip()

    replacements = (
        (" is not ", " is "),
        (" are not ", " are "),
        (" is ", " is not "),
        (" are ", " are not "),
    )
    for old, new in replacements:
        if old in statement:
            return statement.replace(old, new, 1)

    return f"It is not true that {statement}"


def _build_negative_proof(chain: list[str]) -> tuple[str, str]:
    negative_final = _negate_statement(chain[-1])
    negative_chain = [*chain[:-1], negative_final]
    return _format_chain(negative_chain), negative_final


def _flatten_file(hop: int, filename: str) -> list[dict]:
    url = f"{BASE_URL}/{filename}"
    raw_data = _load_json(url)
    rows = []

    for example_id, example in raw_data.items():
        test_example = example["test_example"]
        gold_chain = [
            str(step).strip()
            for step in test_example["chain_of_thought"]
            if str(step).strip()
        ]
        negative_proof, negative_final_statement = _build_negative_proof(gold_chain)
        in_context_examples = []
        for key in sorted(example):
            if not key.startswith("in_context_example"):
                continue
            in_context = example[key]
            in_context_examples.append(
                {
                    "question": in_context["question"],
                    "query": in_context["query"],
                    "chain_of_thought": [
                        str(step).strip()
                        for step in in_context["chain_of_thought"]
                        if str(step).strip()
                    ],
                }
            )

        rows.append(
            {
                "id": f"{hop}hop_{example_id}",
                "source_file": filename,
                "hop": hop,
                "question": test_example["question"],
                "query": test_example["query"],
                "chain_of_thought": gold_chain,
                "gold_proof": _format_chain(gold_chain),
                "gold_final_statement": gold_chain[-1],
                "negative_proof": negative_proof,
                "negative_final_statement": negative_final_statement,
                "in_context_examples": in_context_examples,
            }
        )

    return rows


def load_dataset(**kwargs):
    hops = kwargs.get("hops", sorted(HOP_FILES))
    if isinstance(hops, int):
        hops = [hops]

    rows = []
    for hop in hops:
        hop = int(hop)
        if hop not in HOP_FILES:
            raise ValueError(f"Unsupported PrOntoQA hop count: {hop}")
        rows.extend(_flatten_file(hop, HOP_FILES[hop]))

    return {"test": datasets.Dataset.from_list(rows)}


def doc_to_text(doc: dict) -> str:
    prompt = ""
    for example in doc["in_context_examples"]:
        prompt += (
            f"Q: {example['question']} {example['query']}\n"
            f"A: {_format_chain(example['chain_of_thought'])}\n\n"
        )

    prompt += f"Q: {doc['question']} {doc['query']}\nA:"
    return prompt


def doc_to_target(doc: dict) -> str:
    return " " + doc["gold_proof"]


def doc_to_choice_proof(doc: dict) -> list[str]:
    return [" " + doc["gold_proof"], " " + doc["negative_proof"]]


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    prediction = results[0]
    normalized_prediction = _normalize_text(prediction)
    normalized_gold_proof = _normalize_text(doc["gold_proof"])
    normalized_gold_final = _normalize_text(doc["gold_final_statement"])

    return {
        "final_statement_match": float(normalized_gold_final in normalized_prediction),
        "proof_exact_match": float(normalized_prediction == normalized_gold_proof),
    }
