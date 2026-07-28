"""Validate canonical JSONL datasets before any experiment is run."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from typing import Any, Dict, Optional

from src.data.schemas import SchemaValidationError, validate_document_example
from src.utils.io import ensure_dir, read_jsonl


def _percentile_from_histogram(histogram: Counter, quantile: float) -> Optional[int]:
    total = sum(histogram.values())
    if total == 0:
        return None
    target = max(1, math.ceil(total * quantile))
    cumulative = 0
    for value in sorted(histogram):
        cumulative += histogram[value]
        if cumulative >= target:
            return int(value)
    raise AssertionError("unreachable histogram percentile")


def validate_jsonl(
    path: str,
    expected_split: Optional[str] = None,
    expected_rows: Optional[int] = None,
    expected_dataset_revision: Optional[str] = None,
    expected_dataset_name: Optional[str] = None,
    expected_dataset_fingerprint: Optional[str] = None,
    expected_replacement_rows: Optional[int] = None,
    expected_replacement_characters: Optional[int] = None,
    allow_debug_subset: bool = False,
    allow_replacement_character: bool = False,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "path": path,
        "rows": 0,
        "documents": 0,
        "sentences": 0,
        "references": 0,
        "errors": [],
    }
    seen_ids = set()
    dataset_digest = hashlib.sha256()
    document_counts: Counter = Counter()
    reference_counts: Counter = Counter()
    sentence_counts: Counter = Counter()
    sentence_word_counts: Counter = Counter()
    split_counts: Counter = Counter()
    dataset_names: Counter = Counter()
    dataset_revisions: Counter = Counter()
    total_sentence_words = 0
    replacement_characters = 0
    replacement_character_rows = 0
    debug_subset_rows = 0
    structurally_valid_rows = 0
    for line_number, row in enumerate(read_jsonl(path), start=1):
        report["rows"] += 1
        dataset_digest.update(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        dataset_digest.update(b"\n")
        try:
            validate_document_example(row)
        except (KeyError, SchemaValidationError, TypeError) as exc:
            report["errors"].append({"line": line_number, "error": str(exc)})
            continue

        # Policy violations make the dataset invalid, but must not remove an
        # otherwise well-formed row from health-report denominators.
        structurally_valid_rows += 1
        row_replacement_characters = sum(
            sentence["text"].count("\ufffd")
            for document in row["documents"]
            for section in document["sections"]
            for sentence in section["sentences"]
        ) + sum(ref.count("\ufffd") for ref in row["references"])
        replacement_characters += row_replacement_characters
        if row_replacement_characters:
            replacement_character_rows += 1
            if not allow_replacement_character:
                report["errors"].append(
                    {
                        "line": line_number,
                        "error": (
                            f"row contains {row_replacement_characters} "
                            "Unicode replacement characters"
                        ),
                    }
                )
        if expected_split is not None and row["split"] != expected_split:
            report["errors"].append(
                {
                    "line": line_number,
                    "error": f"split is {row['split']!r}, expected {expected_split!r}",
                }
            )
        if row["id"] in seen_ids:
            report["errors"].append(
                {"line": line_number, "error": f"duplicate example id: {row['id']}"}
            )
        seen_ids.add(row["id"])
        if not row["references"]:
            report["errors"].append(
                {"line": line_number, "error": "benchmark row has no references"}
            )
        metadata = row.get("metadata", {})
        if (
            expected_dataset_revision is not None
            and metadata.get("dataset_revision") != expected_dataset_revision
        ):
            report["errors"].append(
                {
                    "line": line_number,
                    "error": (
                        f"dataset revision is {metadata.get('dataset_revision')!r}, "
                        f"expected {expected_dataset_revision!r}"
                    ),
                }
            )
        if metadata.get("is_debug_subset"):
            debug_subset_rows += 1
            if not allow_debug_subset:
                report["errors"].append(
                    {
                        "line": line_number,
                        "error": "debug-subset row is forbidden in a formal dataset validation",
                    }
                )

        report["documents"] += len(row["documents"])
        document_counts[len(row["documents"])] += 1
        reference_counts[len(row["references"])] += 1
        split_counts[row["split"]] += 1
        if row.get("dataset_name"):
            dataset_names[str(row["dataset_name"])] += 1
        if metadata.get("dataset_revision"):
            dataset_revisions[str(metadata["dataset_revision"])] += 1
        row_sentence_count = 0
        for document in row["documents"]:
            for section in document["sections"]:
                for sentence in section["sentences"]:
                    report["sentences"] += 1
                    row_sentence_count += 1
                    word_count = len(sentence["text"].split())
                    sentence_word_counts[word_count] += 1
                    total_sentence_words += word_count
        sentence_counts[row_sentence_count] += 1
        report["references"] += len(row["references"])
    if expected_rows is not None and report["rows"] != expected_rows:
        report["errors"].append(
            {
                "line": None,
                "error": f"row count is {report['rows']}, expected {expected_rows}",
            }
        )
    if len(dataset_revisions) > 1:
        report["errors"].append(
            {"line": None, "error": "dataset mixes multiple source revisions"}
        )
    if expected_dataset_name is not None:
        unexpected_names = {
            name: count
            for name, count in dataset_names.items()
            if name != expected_dataset_name
        }
        if unexpected_names or sum(dataset_names.values()) != structurally_valid_rows:
            report["errors"].append(
                {
                    "line": None,
                    "error": (
                        f"dataset names are {dict(sorted(dataset_names.items()))!r}, "
                        f"expected every row to be {expected_dataset_name!r}"
                    ),
                }
            )
    actual_fingerprint = dataset_digest.hexdigest()
    if (
        expected_dataset_fingerprint is not None
        and actual_fingerprint != expected_dataset_fingerprint
    ):
        report["errors"].append(
            {
                "line": None,
                "error": (
                    f"dataset fingerprint is {actual_fingerprint}, expected "
                    f"{expected_dataset_fingerprint}"
                ),
            }
        )
    if (
        expected_replacement_rows is not None
        and replacement_character_rows != expected_replacement_rows
    ):
        report["errors"].append(
            {
                "line": None,
                "error": (
                    f"Unicode replacement row count is {replacement_character_rows}, "
                    f"expected {expected_replacement_rows}"
                ),
            }
        )
    if (
        expected_replacement_characters is not None
        and replacement_characters != expected_replacement_characters
    ):
        report["errors"].append(
            {
                "line": None,
                "error": (
                    "Unicode replacement character count is "
                    f"{replacement_characters}, expected "
                    f"{expected_replacement_characters}"
                ),
            }
        )
    report["valid"] = not report["errors"] and report["rows"] > 0
    report["dataset_fingerprint"] = actual_fingerprint
    report["health"] = {
        "split_counts": dict(sorted(split_counts.items())),
        "dataset_names": dict(sorted(dataset_names.items())),
        "dataset_revisions": dict(sorted(dataset_revisions.items())),
        "structurally_valid_rows": structurally_valid_rows,
        "documents_per_example": dict(sorted(document_counts.items())),
        "references_per_example": dict(sorted(reference_counts.items())),
        "sentences_per_example": {
            "mean": (
                report["sentences"] / structurally_valid_rows
                if structurally_valid_rows
                else None
            ),
            "min": min(sentence_counts, default=None),
            "p50": _percentile_from_histogram(sentence_counts, 0.50),
            "p95": _percentile_from_histogram(sentence_counts, 0.95),
            "p99": _percentile_from_histogram(sentence_counts, 0.99),
            "max": max(sentence_counts, default=None),
            "under_20": sum(
                count for sentence_count, count in sentence_counts.items()
                if sentence_count < 20
            ),
            "under_40": sum(
                count for sentence_count, count in sentence_counts.items()
                if sentence_count < 40
            ),
        },
        "sentence_words": {
            "mean": (
                total_sentence_words / report["sentences"] if report["sentences"] else None
            ),
            "p50": _percentile_from_histogram(sentence_word_counts, 0.50),
            "p95": _percentile_from_histogram(sentence_word_counts, 0.95),
            "p99": _percentile_from_histogram(sentence_word_counts, 0.99),
            "max": max(sentence_word_counts, default=None),
            "over_80": sum(
                count for word_count, count in sentence_word_counts.items() if word_count > 80
            ),
        },
        "unicode_replacement_characters": replacement_characters,
        "unicode_replacement_rows": replacement_character_rows,
        "debug_subset_rows": debug_subset_rows,
    }
    if report["rows"] == 0:
        report["errors"].append({"line": None, "error": "dataset is empty"})
        report["valid"] = False
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="canonical JSONL dataset")
    parser.add_argument("--split", default=None, help="expected train/validation/test split")
    parser.add_argument("--expected_rows", type=int, default=None)
    parser.add_argument("--expected_dataset_revision", default=None)
    parser.add_argument("--expected_dataset_name", default=None)
    parser.add_argument("--expected_dataset_fingerprint", default=None)
    parser.add_argument("--expected_replacement_rows", type=int, default=None)
    parser.add_argument("--expected_replacement_characters", type=int, default=None)
    parser.add_argument("--allow_debug_subset", action="store_true")
    parser.add_argument("--allow_replacement_character", action="store_true")
    parser.add_argument("--report_out", default=None, help="optional JSON report path")
    args = parser.parse_args()

    report = validate_jsonl(
        args.input,
        expected_split=args.split,
        expected_rows=args.expected_rows,
        expected_dataset_revision=args.expected_dataset_revision,
        expected_dataset_name=args.expected_dataset_name,
        expected_dataset_fingerprint=args.expected_dataset_fingerprint,
        expected_replacement_rows=args.expected_replacement_rows,
        expected_replacement_characters=args.expected_replacement_characters,
        allow_debug_subset=args.allow_debug_subset,
        allow_replacement_character=args.allow_replacement_character,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report_out:
        ensure_dir(os.path.dirname(args.report_out) or ".")
        with open(args.report_out, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
