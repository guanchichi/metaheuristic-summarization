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
    sentence_word_counts: Counter = Counter()
    split_counts: Counter = Counter()
    dataset_revisions: Counter = Counter()
    total_sentence_words = 0
    replacement_characters = 0
    debug_subset_rows = 0
    for line_number, row in enumerate(read_jsonl(path), start=1):
        report["rows"] += 1
        dataset_digest.update(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        dataset_digest.update(b"\n")
        try:
            validate_document_example(row)
            row_replacement_characters = sum(
                sentence["text"].count("\ufffd")
                for document in row["documents"]
                for section in document["sections"]
                for sentence in section["sentences"]
            ) + sum(ref.count("\ufffd") for ref in row["references"])
            replacement_characters += row_replacement_characters
            if row_replacement_characters and not allow_replacement_character:
                raise SchemaValidationError(
                    f"row contains {row_replacement_characters} Unicode replacement characters"
                )
            if expected_split is not None and row["split"] != expected_split:
                raise SchemaValidationError(
                    f"split is {row['split']!r}, expected {expected_split!r}"
                )
            if row["id"] in seen_ids:
                raise SchemaValidationError(f"duplicate example id: {row['id']}")
            if not row["references"]:
                raise SchemaValidationError("benchmark row has no references")
            metadata = row.get("metadata", {})
            if (
                expected_dataset_revision is not None
                and metadata.get("dataset_revision") != expected_dataset_revision
            ):
                raise SchemaValidationError(
                    f"dataset revision is {metadata.get('dataset_revision')!r}, "
                    f"expected {expected_dataset_revision!r}"
                )
            if metadata.get("is_debug_subset"):
                debug_subset_rows += 1
                if not allow_debug_subset:
                    raise SchemaValidationError(
                        "debug-subset row is forbidden in a formal dataset validation"
                    )
            seen_ids.add(row["id"])
            report["documents"] += len(row["documents"])
            document_counts[len(row["documents"])] += 1
            reference_counts[len(row["references"])] += 1
            split_counts[row["split"]] += 1
            if metadata.get("dataset_revision"):
                dataset_revisions[str(metadata["dataset_revision"])] += 1
            for document in row["documents"]:
                for section in document["sections"]:
                    for sentence in section["sentences"]:
                        report["sentences"] += 1
                        word_count = len(sentence["text"].split())
                        sentence_word_counts[word_count] += 1
                        total_sentence_words += word_count
            report["references"] += len(row["references"])
        except (KeyError, SchemaValidationError, TypeError) as exc:
            report["errors"].append({"line": line_number, "error": str(exc)})
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
    report["valid"] = not report["errors"] and report["rows"] > 0
    report["dataset_fingerprint"] = dataset_digest.hexdigest()
    report["health"] = {
        "split_counts": dict(sorted(split_counts.items())),
        "dataset_revisions": dict(sorted(dataset_revisions.items())),
        "documents_per_example": dict(sorted(document_counts.items())),
        "references_per_example": dict(sorted(reference_counts.items())),
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
    parser.add_argument("--allow_debug_subset", action="store_true")
    parser.add_argument("--allow_replacement_character", action="store_true")
    parser.add_argument("--report_out", default=None, help="optional JSON report path")
    args = parser.parse_args()

    report = validate_jsonl(
        args.input,
        expected_split=args.split,
        expected_rows=args.expected_rows,
        expected_dataset_revision=args.expected_dataset_revision,
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
