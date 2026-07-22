"""Dataset loader for the OLMo-compatible RuleTaker evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import datasets


RULETAKER_URL = (
    "https://aristo-data-public.s3-us-west-2.amazonaws.com/ruletaker/"
    "rule-reasoning-dataset-V2020.2.5.zip"
)
ARCHIVE_PREFIX = Path("rule-reasoning-dataset-V2020.2.5.0") / "original"
SUPPORTED_DEPTHS = (0, 1, 2, 3, 5)
EXPECTED_SHA256 = {
    0: "666d84ae6d1da0cb82bc86945b2ff6bfd53d2156f599fd7dd81569b3b3f1bc7a",
    1: "140dc5166bafd4e6ad5597db49942b1c45ca2925d33fef9dcf5098319bba0494",
    2: "da3129e2e46c896c2e710f95face897a85754621b3fd90cfd967a2934227cdba",
    3: "b91197f1ce78c31090a7a530db4b315725ece048f8eb715486e267c5a8569455",
    5: "917e61f2349b0d6e0a31cb8ad3132f4affff5795d91095dea2a1aad0d31f8801",
}


def _normalize_depth(depth: int | str) -> int:
    if isinstance(depth, str):
        match = re.fullmatch(r"(?:depth-?|d)?(\d+)", depth.strip().lower())
        if match is None:
            raise ValueError(f"Invalid RuleTaker depth: {depth!r}")
        depth = int(match.group(1))
    depth = int(depth)
    if depth not in SUPPORTED_DEPTHS:
        raise ValueError(
            f"Unsupported RuleTaker depth {depth}; expected one of {SUPPORTED_DEPTHS}"
        )
    return depth


def _item_sort_key(key: str) -> tuple[str, int]:
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", str(key))
    if match:
        return match.group(1), int(match.group(2))
    return str(key), 0


def _build_theory(record: dict[str, Any]) -> str:
    if record.get("theory"):
        return str(record["theory"])

    parts = []
    for field in ("triples", "rules"):
        items = record.get(field) or {}
        for key in sorted(items, key=_item_sort_key):
            text = items[key].get("text")
            if text:
                parts.append(str(text))
    return " ".join(parts)


def _normalize_label(answer: bool | str) -> int:
    if isinstance(answer, bool):
        return 0 if answer else 1
    normalized = str(answer).strip().lower()
    if normalized == "true":
        return 0
    if normalized == "false":
        return 1
    raise ValueError(f"Unexpected RuleTaker answer: {answer!r}")


def _flatten_records(records: list[dict[str, Any]], depth: int) -> list[dict[str, Any]]:
    rows = []
    for record_index, record in enumerate(records):
        theory = _build_theory(record)
        questions = record.get("questions") or {}
        for question_key in sorted(questions, key=_item_sort_key):
            question = questions[question_key]
            rows.append(
                {
                    "id": str(
                        question.get("id")
                        or f"depth-{depth}:{record_index}:{question_key}"
                    ),
                    "theory": theory,
                    "question": str(question["question"]),
                    "label": _normalize_label(question["answer"]),
                    "depth": depth,
                }
            )
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_data_file(root: Path, depth: int) -> Path:
    filename = "meta-test.jsonl"
    depth_dir = f"depth-{depth}"

    if root.is_file():
        if root.name == filename and root.parent.name == depth_dir:
            return root
        raise FileNotFoundError(
            f"Expected a RuleTaker {filename} file, got {root}"
        )

    candidates = (
        root / ARCHIVE_PREFIX / depth_dir / filename,
        root / "original" / depth_dir / filename,
        root / depth_dir / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # Some datasets versions preserve an additional cache/archive directory
    # around the zip contents. Fall back to a recursive search, but require the
    # official ``original/depth-N`` suffix so that the archive's problog and
    # language variants cannot be selected accidentally.
    recursive_matches = sorted(
        candidate
        for candidate in root.rglob(filename)
        if candidate.parent.name == depth_dir
        and candidate.parent.parent.name == "original"
        and "__MACOSX" not in candidate.parts
    )
    if len(recursive_matches) == 1:
        return recursive_matches[0]
    if len(recursive_matches) > 1:
        matches = ", ".join(str(path) for path in recursive_matches)
        raise FileNotFoundError(
            f"Multiple RuleTaker depth-{depth} {filename} files found below "
            f"{root}: {matches}"
        )

    raise FileNotFoundError(
        f"RuleTaker depth-{depth} {filename} not found below {root}. "
        "The archive may be only partially extracted; remove this extracted "
        "cache directory and retry once before launching distributed evaluation."
    )


def load_dataset(
    depth: int | str = 3,
    data_dir: str | None = None,
    verify_checksum: bool = True,
    **_: Any,
) -> dict[str, datasets.Dataset]:
    """Load and flatten one RuleTaker ``original/meta-test`` depth.

    By default the official V2020.2.5 archive is downloaded through the
    Hugging Face datasets cache. ``data_dir`` may instead point at an already
    extracted archive, its ``original`` directory, or a directory containing
    ``depth-N/meta-test.jsonl``.
    """

    normalized_depth = _normalize_depth(depth)
    if data_dir is None:
        extracted = datasets.DownloadManager().download_and_extract(RULETAKER_URL)
        root = Path(extracted)
    else:
        root = Path(data_dir).expanduser().resolve()

    data_file = _find_data_file(root, normalized_depth)
    if verify_checksum:
        actual_digest = _sha256(data_file)
        expected_digest = EXPECTED_SHA256[normalized_depth]
        if actual_digest != expected_digest:
            raise ValueError(
                f"Checksum mismatch for {data_file}: expected {expected_digest}, "
                f"got {actual_digest}"
            )

    with data_file.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    rows = _flatten_records(records, normalized_depth)
    if not rows:
        raise ValueError(f"No RuleTaker questions found in {data_file}")
    return {"test": datasets.Dataset.from_list(rows)}
