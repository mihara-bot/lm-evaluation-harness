import json
import re
from typing import Iterable, Optional
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


def _split_statements(text: str) -> list[str]:
    return [f"{statement.strip()}." for statement in str(text).split(".") if statement.strip()]


def _parse_entity_statement(statement: str) -> Optional[tuple[str, str]]:
    match = re.match(r"^(.+?) is (.+)\.$", str(statement).strip())
    if not match:
        return None
    return match.group(1), match.group(2)


def _predicate_kind(predicate: str) -> str:
    predicate = str(predicate).strip()
    if predicate.startswith(("a ", "an ")):
        return "class"
    if predicate.startswith("not "):
        return "neg_property"
    return "property"


def _select_counterfactual_step(
    question: str, chain: list[str], target_index: int
) -> Optional[str]:
    target = chain[target_index]
    parsed_target = _parse_entity_statement(target)
    if parsed_target is None:
        return None

    subject, target_predicate = parsed_target
    normalized_chain = {_normalize_text(step) for step in chain}
    target_kind = _predicate_kind(target_predicate)

    same_subject_facts = []
    predicate_pool = []
    for statement in _split_statements(question):
        parsed_statement = _parse_entity_statement(statement)
        if parsed_statement is None:
            continue
        statement_subject, statement_predicate = parsed_statement
        if _normalize_text(statement) not in normalized_chain:
            predicate_pool.append(statement_predicate)
            if statement_subject == subject:
                same_subject_facts.append(statement)

    def sort_key(statement: str) -> tuple[int, int, str]:
        parsed_statement = _parse_entity_statement(statement)
        predicate = parsed_statement[1] if parsed_statement else ""
        return (
            int(_predicate_kind(predicate) != target_kind),
            abs(len(statement) - len(target)),
            statement,
        )

    if same_subject_facts:
        return min(same_subject_facts, key=sort_key)

    for statement in chain:
        parsed_statement = _parse_entity_statement(statement)
        if parsed_statement is not None:
            predicate_pool.append(parsed_statement[1])

    synthetic_steps = []
    for predicate in predicate_pool:
        if _normalize_text(predicate) == _normalize_text(target_predicate):
            continue
        synthetic_step = f"{subject} is {predicate}."
        if _normalize_text(synthetic_step) not in normalized_chain:
            synthetic_steps.append(synthetic_step)

    if synthetic_steps:
        return min(synthetic_steps, key=sort_key)

    return None


def _select_distractor_statement(
    question: str, chain: list[str], target_index: int
) -> Optional[str]:
    target = chain[target_index]
    chain_statements = {_normalize_text(step) for step in chain}
    candidates = [
        statement
        for statement in _split_statements(question)
        if _normalize_text(statement) not in chain_statements
    ]
    if not candidates:
        return None

    return min(candidates, key=lambda statement: (abs(len(statement) - len(target)), statement))


def _build_negative_proofs(question: str, chain: list[str]) -> list[dict[str, str]]:
    negatives = []
    seen_proofs = {_format_chain(chain)}

    for target_index in range(0, max(1, len(chain) - 1), 2):
        if target_index >= len(chain) - 1:
            continue

        original_step = chain[target_index]
        replacement_step = _select_counterfactual_step(question, chain, target_index)
        if replacement_step is None:
            replacement_step = _select_distractor_statement(question, chain, target_index)
        if replacement_step is None:
            continue

        negative_chain = [*chain]
        negative_chain[target_index] = replacement_step
        negative_proof = _format_chain(negative_chain)
        normalized_negative_proof = _normalize_text(negative_proof)
        if normalized_negative_proof in seen_proofs:
            continue

        seen_proofs.add(normalized_negative_proof)
        negatives.append(
            {
                "proof": negative_proof,
                "final_statement": chain[-1],
                "original_step": original_step,
                "replacement_step": replacement_step,
            }
        )

    return negatives


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
        negative_records = _build_negative_proofs(test_example["question"], gold_chain)
        if not negative_records:
            raise ValueError(f"Could not build PrOntoQA hard negative for {example_id}")
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
                "negative_proof": negative_records[0]["proof"],
                "negative_final_statement": negative_records[0]["final_statement"],
                "negative_original_step": negative_records[0]["original_step"],
                "negative_replacement_step": negative_records[0]["replacement_step"],
                "negative_proofs": [record["proof"] for record in negative_records],
                "negative_final_statements": [
                    record["final_statement"] for record in negative_records
                ],
                "negative_original_steps": [
                    record["original_step"] for record in negative_records
                ],
                "negative_replacement_steps": [
                    record["replacement_step"] for record in negative_records
                ],
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
    negative_proofs = doc.get("negative_proofs") or [doc["negative_proof"]]
    return [" " + doc["gold_proof"]] + [" " + proof for proof in negative_proofs]


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    prediction = results[0]
    normalized_prediction = _normalize_text(prediction)
    normalized_gold_proof = _normalize_text(doc["gold_proof"])
    normalized_gold_final = _normalize_text(doc["gold_final_statement"])

    return {
        "final_statement_match": float(normalized_gold_final in normalized_prediction),
        "proof_exact_match": float(normalized_prediction == normalized_gold_proof),
    }
