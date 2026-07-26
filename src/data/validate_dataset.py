"""Validate canonical JSONL datasets before any experiment is run."""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any, Dict, Optional

from src.data.schemas import SchemaValidationError, validate_document_example
from src.utils.io import read_jsonl


def validate_jsonl(path: str, expected_split: Optional[str] = None) -> Dict[str, Any]:
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
    for line_number, row in enumerate(read_jsonl(path), start=1):
        report["rows"] += 1
        dataset_digest.update(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        dataset_digest.update(b"\n")
        try:
            validate_document_example(row)
            if expected_split is not None and row["split"] != expected_split:
                raise SchemaValidationError(
                    f"split is {row['split']!r}, expected {expected_split!r}"
                )
            if row["id"] in seen_ids:
                raise SchemaValidationError(f"duplicate example id: {row['id']}")
            seen_ids.add(row["id"])
            report["documents"] += len(row["documents"])
            report["sentences"] += sum(
                len(section["sentences"])
                for document in row["documents"]
                for section in document["sections"]
            )
            report["references"] += len(row["references"])
        except (KeyError, SchemaValidationError, TypeError) as exc:
            report["errors"].append({"line": line_number, "error": str(exc)})
    report["valid"] = not report["errors"] and report["rows"] > 0
    report["dataset_fingerprint"] = dataset_digest.hexdigest()
    if report["rows"] == 0:
        report["errors"].append({"line": None, "error": "dataset is empty"})
        report["valid"] = False
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="canonical JSONL dataset")
    parser.add_argument("--split", default=None, help="expected train/validation/test split")
    args = parser.parse_args()

    report = validate_jsonl(args.input, expected_split=args.split)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
