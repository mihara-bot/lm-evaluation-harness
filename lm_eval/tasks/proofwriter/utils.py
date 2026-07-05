import datasets


CHOICES = ["True", "False", "Unknown"]
ANSWER_TO_LABEL = {answer.lower(): idx for idx, answer in enumerate(CHOICES)}


def _normalize_answer(answer: str) -> str:
    return str(answer).strip().lower()


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
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

    return dataset.map(_process_doc)
