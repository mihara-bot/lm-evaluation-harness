import json
from urllib.request import urlopen

import datasets


DATA_URL = "https://raw.githubusercontent.com/google-deepmind/bbeh/main/bbeh/mini/data.json"
_RAW_DATA_CACHE = {}
MINI_INDICES_BY_TASK = {
    "bbeh_boardgame_qa": (
        23, 45, 72, 75, 96, 97, 127, 128, 139, 165,
        170, 252, 292, 293, 302, 318, 344, 354, 400, 408,
    ),
    "bbeh_boolean_expressions": (
        13, 38, 46, 67, 102, 125, 148, 150, 161, 212,
        224, 236, 261, 278, 306, 349, 392, 410, 427, 456,
    ),
    "bbeh_buggy_tables": (
        43, 52, 63, 92, 107, 154, 197, 199, 202, 253,
        324, 327, 348, 379, 383, 384, 407, 422, 429, 439,
    ),
    "bbeh_causal_understanding": (
        12, 54, 61, 64, 103, 135, 138, 146, 218, 237,
        243, 256, 298, 308, 345, 346, 374, 387, 431, 443,
    ),
    "bbeh_disambiguation_qa": (
        11, 65, 81, 113, 132, 156, 168, 220, 229, 241,
        259, 260, 263, 274, 291, 313, 355, 356, 357, 391,
    ),
    "bbeh_dyck_languages": (
        2, 5, 42, 49, 86, 117, 118, 137, 141, 166,
        195, 207, 232, 267, 287, 315, 336, 389, 416, 445,
    ),
    "bbeh_geometric_shapes": (
        10, 20, 41, 94, 101, 122, 131, 142, 169, 175,
        226, 245, 317, 337, 359, 388, 396, 411, 436, 455,
    ),
    "bbeh_hyperbaton": (
        32, 40, 76, 112, 143, 153, 203, 215, 219, 235,
        247, 332, 339, 358, 366, 376, 378, 430, 442, 449,
    ),
    "bbeh_linguini": (
        14, 34, 88, 121, 123, 157, 201, 217, 234, 238,
        250, 258, 276, 321, 361, 382, 394, 397, 419, 425,
    ),
    "bbeh_movie_recommendation": (
        27, 53, 59, 74, 77, 116, 129, 149, 158, 176,
        177, 183, 208, 222, 295, 314, 328, 362, 421, 447,
    ),
    "bbeh_multistep_arithmetic": (
        6, 16, 30, 33, 62, 69, 70, 93, 109, 140,
        160, 167, 186, 283, 311, 334, 341, 405, 452, 457,
    ),
    "bbeh_nycc": (
        15, 17, 24, 31, 111, 115, 130, 134, 145, 159,
        173, 228, 244, 271, 275, 331, 365, 385, 413, 420,
    ),
    "bbeh_object_counting": (
        9, 48, 68, 80, 108, 126, 191, 198, 206, 264,
        272, 288, 323, 350, 380, 398, 424, 444, 448, 453,
    ),
    "bbeh_object_properties": (
        21, 37, 66, 84, 89, 172, 209, 211, 230, 246,
        249, 304, 338, 340, 371, 381, 386, 412, 435, 454,
    ),
    "bbeh_sarc_triples": (
        39, 57, 71, 110, 185, 192, 227, 231, 240, 266,
        268, 285, 305, 333, 360, 390, 409, 415, 423, 446,
    ),
    "bbeh_shuffled_objects": (
        7, 19, 26, 47, 58, 78, 98, 164, 174, 189,
        282, 286, 307, 335, 363, 367, 399, 418, 432, 438,
    ),
    "bbeh_spatial_reasoning": (
        0, 90, 91, 124, 144, 155, 162, 188, 194, 233,
        239, 242, 248, 251, 289, 316, 329, 352, 401, 417,
    ),
    "bbeh_sportqa": (
        3, 18, 28, 35, 55, 73, 95, 114, 178, 179,
        221, 280, 281, 290, 297, 300, 325, 347, 402, 426,
    ),
    "bbeh_temporal_sequence": (
        50, 79, 82, 100, 105, 182, 214, 262, 269, 277,
        296, 301, 312, 322, 342, 393, 404, 406, 428, 434,
    ),
    "bbeh_time_arithmetic": (
        29, 36, 85, 106, 119, 196, 200, 279, 299, 319,
        320, 326, 330, 343, 353, 364, 368, 372, 440, 459,
    ),
    "bbeh_web_of_lies": (
        8, 22, 51, 56, 60, 104, 133, 171, 181, 187,
        204, 254, 255, 310, 351, 370, 377, 433, 437, 451,
    ),
    "bbeh_word_sorting": (
        83, 87, 99, 147, 163, 180, 184, 193, 210, 213,
        257, 273, 294, 303, 309, 369, 373, 375, 395, 450,
    ),
    "bbeh_zebra_puzzles": (
        1, 4, 25, 44, 120, 136, 151, 152, 190, 205,
        216, 223, 225, 265, 270, 284, 403, 414, 441, 458,
    ),
}
MINI_TASK_BY_INDEX = {
    idx: task_name
    for task_name, indices in MINI_INDICES_BY_TASK.items()
    for idx in indices
}


def strip_latex(response: str) -> str:
    if response.startswith("$") and response.endswith("$"):
        response = response[1:-1]
    if "boxed{" in response and response.endswith("}"):
        response = response[0:-1].split("boxed{")[1]
    if "text{" in response and response.endswith("}"):
        response = response[0:-1].split("text{")[1]
    if "texttt{" in response and response.endswith("}"):
        response = response[0:-1].split("texttt{")[1]
    return response


def extract_answer(sample: str) -> str:
    answer_prefixes = [
        "The answer is:",
        "The final answer is ",
        "The final answer is: ",
        "The answer is ",
    ]
    answer = sample
    for answer_prefix in answer_prefixes:
        if answer_prefix in answer:
            answer = answer.split(answer_prefix)[-1].strip()
    if answer.endswith("."):
        answer = answer[:-1]
    return strip_latex(answer)


def fuzzy_match(prediction: str, reference: str) -> bool:
    if prediction == reference:
        return True

    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        return reference[1] == prediction

    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass

    if prediction.replace("'", "") == reference.replace("'", ""):
        return True

    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True

    if prediction.endswith("?") and prediction[:-1] == reference:
        return True

    return False


def preprocess_sample(sample: str) -> str:
    prediction = extract_answer(str(sample).strip()).lower()
    prediction = prediction.replace(", ", ",").replace("**", "")
    prediction = prediction.split("\n")[0]
    prediction = prediction[0:-1] if prediction.endswith(".") else prediction
    return prediction


def preprocess_reference(reference: str) -> str:
    reference = str(reference).strip().lower()
    reference = reference.replace(", ", ",")
    return reference


def evaluate_correctness(sample: str, reference: str) -> bool:
    prediction = preprocess_sample(sample)
    reference = preprocess_reference(reference)
    return fuzzy_match(prediction, reference)


def _load_json(url: str) -> dict:
    if url in _RAW_DATA_CACHE:
        return _RAW_DATA_CACHE[url]
    with urlopen(url, timeout=60) as response:
        raw_data = json.loads(response.read().decode("utf-8"))
    _RAW_DATA_CACHE[url] = raw_data
    return raw_data


def load_dataset(url: str = DATA_URL, task_name: str = None, version=None, **kwargs):
    if task_name is not None and task_name not in MINI_INDICES_BY_TASK:
        raise ValueError(f"Unknown BBEH mini task_name: {task_name}")

    raw_data = _load_json(url)
    rows = []
    for idx, example in enumerate(raw_data["examples"]):
        example_task_name = MINI_TASK_BY_INDEX[idx]
        if task_name is not None and example_task_name != task_name:
            continue
        rows.append(
            {
                "id": f"bbeh_mini_{idx}",
                "task_name": example_task_name,
                "input": example["input"],
                "target": example["target"],
            }
        )
    return {"test": datasets.Dataset.from_list(rows)}


def doc_to_text(doc: dict) -> str:
    return f"{doc['input'].strip()}\n\nThe final answer is:"


def doc_to_target(doc: dict) -> str:
    return str(doc["target"])


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    return {"exact_match": float(evaluate_correctness(results[0], doc["target"]))}
