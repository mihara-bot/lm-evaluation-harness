"""Document filters for OLMo iGSM-Easy Arithmetic tasks."""


SUPPORTED_DEPTHS = (2, 3, 4)


def process_docs_by_depth(dataset, depth: int):
    """Select the fixed test examples for one target dependency depth."""

    if depth not in SUPPORTED_DEPTHS:
        raise ValueError(
            f"Unsupported iGSM arithmetic depth {depth}; expected one of "
            f"{SUPPORTED_DEPTHS}"
        )
    return dataset.filter(lambda doc: int(doc["target_depth"]) == depth)


def process_docs_d2(dataset):
    return process_docs_by_depth(dataset, 2)


def process_docs_d3(dataset):
    return process_docs_by_depth(dataset, 3)


def process_docs_d4(dataset):
    return process_docs_by_depth(dataset, 4)
