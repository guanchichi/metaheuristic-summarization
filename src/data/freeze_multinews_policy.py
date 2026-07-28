"""Freeze the Multi-News validation data-quality policy before experiments.

The main analysis preserves every structurally valid official row, including
rows containing U+FFFD.  A paired clean sensitivity artifact excludes exactly
the rows named in a tracked manifest.  No text is repaired or rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from typing import Any, Dict, Iterable, Tuple

from src.data.preprocess_multinews import (
    DATASET_ID,
    DATASET_REVISION,
    PREPROCESSOR_VERSION,
)
from src.data.policy import load_frozen_policy, sha256_file, validate_dataset_policy_request
from src.data.validate_dataset import validate_jsonl
from src.utils.io import ensure_dir, read_jsonl, write_jsonl_atomic


POLICY_ID = "multinews-validation-v1"
EXPECTED_MAIN_ROWS = 5621
EXPECTED_REPLACEMENT_ROWS = 72
EXPECTED_REPLACEMENT_CHARACTERS = 1042
EXPECTED_EXCLUDED_SOURCE_ROW = 4850


def replacement_character_counts(row: Dict[str, Any]) -> Tuple[int, int]:
    source_count = sum(
        sentence["text"].count("\ufffd")
        for document in row["documents"]
        for section in document["sections"]
        for sentence in section["sentences"]
    )
    reference_count = sum(reference.count("\ufffd") for reference in row["references"])
    return source_count, reference_count


def _write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    output_dir = os.path.dirname(path) or "."
    ensure_dir(output_dir)
    handle, temporary_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".",
        suffix=".partial",
        dir=output_dir,
        text=True,
    )
    os.close(handle)
    try:
        with open(
            temporary_path, "w", encoding="utf-8", newline="\n"
        ) as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _load_canonical_exclusions(path: str | None) -> Dict[str, Any]:
    if not path:
        return {
            "expected_rows": 1,
            "source_row_indices": [EXPECTED_EXCLUDED_SOURCE_ROW],
            "reason": "empty source cluster",
            "manifest_path": None,
            "manifest_file_sha256": None,
        }
    exclusions = list(read_jsonl(path))
    indices = [int(row["source_row_index"]) for row in exclusions]
    if indices != [EXPECTED_EXCLUDED_SOURCE_ROW]:
        raise ValueError(
            "canonical exclusion manifest must contain exactly source row "
            f"{EXPECTED_EXCLUDED_SOURCE_ROW}, got {indices}"
        )
    return {
        "expected_rows": len(exclusions),
        "source_row_indices": indices,
        "reason": exclusions[0]["reason"],
        "manifest_path": path.replace("\\", "/"),
        "manifest_file_sha256": sha256_file(path),
    }


def freeze_policy(
    *,
    input_path: str,
    clean_output_path: str,
    replacement_manifest_path: str,
    policy_output_path: str,
    exclusion_manifest_path: str | None = None,
    expected_main_rows: int = EXPECTED_MAIN_ROWS,
    expected_replacement_rows: int = EXPECTED_REPLACEMENT_ROWS,
    expected_replacement_characters: int = EXPECTED_REPLACEMENT_CHARACTERS,
) -> Dict[str, Any]:
    """Generate and verify the paired clean artifact and frozen policy."""

    if os.path.abspath(input_path) == os.path.abspath(clean_output_path):
        raise ValueError("clean sensitivity output must differ from the main input")

    main_report = validate_jsonl(
        input_path,
        expected_split="validation",
        expected_rows=expected_main_rows,
        expected_dataset_revision=DATASET_REVISION,
        expected_dataset_name="Multi-News",
        expected_replacement_rows=expected_replacement_rows,
        expected_replacement_characters=expected_replacement_characters,
        allow_replacement_character=True,
    )
    if not main_report["valid"]:
        raise ValueError(f"main canonical dataset failed policy precheck: {main_report['errors']}")

    replacement_manifest: list[Dict[str, Any]] = []
    clean_rows_written = 0

    def clean_rows() -> Iterable[Dict[str, Any]]:
        nonlocal clean_rows_written
        for line_number, row in enumerate(read_jsonl(input_path), start=1):
            source_count, reference_count = replacement_character_counts(row)
            total_count = source_count + reference_count
            if total_count:
                replacement_manifest.append(
                    {
                        "id": row["id"],
                        "source_row_index": row.get("metadata", {}).get(
                            "source_row_index"
                        ),
                        "canonical_line_number": line_number,
                        "data_fingerprint": row["data_fingerprint"],
                        "source_replacement_characters": source_count,
                        "reference_replacement_characters": reference_count,
                        "total_replacement_characters": total_count,
                        "sensitivity_action": "exclude_row_without_text_repair",
                    }
                )
                continue
            clean_rows_written += 1
            yield row

    write_jsonl_atomic(clean_output_path, clean_rows())
    write_jsonl_atomic(replacement_manifest_path, replacement_manifest)

    if len(replacement_manifest) != expected_replacement_rows:
        raise ValueError(
            f"replacement manifest has {len(replacement_manifest)} rows, "
            f"expected {expected_replacement_rows}"
        )
    observed_characters = sum(
        row["total_replacement_characters"] for row in replacement_manifest
    )
    if observed_characters != expected_replacement_characters:
        raise ValueError(
            f"replacement manifest has {observed_characters} characters, "
            f"expected {expected_replacement_characters}"
        )
    expected_clean_rows = expected_main_rows - expected_replacement_rows
    if clean_rows_written != expected_clean_rows:
        raise ValueError(
            f"clean sensitivity has {clean_rows_written} rows, "
            f"expected {expected_clean_rows}"
        )

    clean_report = validate_jsonl(
        clean_output_path,
        expected_split="validation",
        expected_rows=expected_clean_rows,
        expected_dataset_revision=DATASET_REVISION,
        expected_dataset_name="Multi-News",
        expected_replacement_rows=0,
        expected_replacement_characters=0,
        allow_replacement_character=False,
    )
    if not clean_report["valid"]:
        raise ValueError(
            f"clean sensitivity artifact failed postcheck: {clean_report['errors']}"
        )

    removed_ids_digest = hashlib.sha256()
    for row in replacement_manifest:
        removed_ids_digest.update(row["id"].encode("utf-8"))
        removed_ids_digest.update(b"\n")

    policy = {
        "policy_schema_version": "1.0",
        "policy_id": POLICY_ID,
        "status": "frozen_before_validation_results",
        "dataset": {
            "name": "Multi-News",
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "preprocessor_version": PREPROCESSOR_VERSION,
            "split": "validation",
        },
        "canonical_exclusions": _load_canonical_exclusions(exclusion_manifest_path),
        "replacement_character_manifest": {
            "path": replacement_manifest_path.replace("\\", "/"),
            "rows": len(replacement_manifest),
            "characters": observed_characters,
            "row_ids_sha256": removed_ids_digest.hexdigest(),
            "file_sha256": sha256_file(replacement_manifest_path),
        },
        "analyses": {
            "main": {
                "role": "primary_validation",
                "artifact_path": input_path.replace("\\", "/"),
                "row_policy": "include_replacement_rows_unchanged",
                "text_repair": "forbidden",
                "expected_rows": main_report["rows"],
                "expected_dataset_fingerprint": main_report["dataset_fingerprint"],
                "expected_file_sha256": sha256_file(input_path),
                "expected_replacement_rows": expected_replacement_rows,
                "expected_replacement_characters": expected_replacement_characters,
                "allow_replacement_character": True,
            },
            "clean_sensitivity": {
                "role": "paired_data_quality_sensitivity",
                "artifact_path": clean_output_path.replace("\\", "/"),
                "row_policy": "exclude_exact_manifest_rows_without_text_repair",
                "text_repair": "forbidden",
                "expected_rows": clean_report["rows"],
                "expected_dataset_fingerprint": clean_report[
                    "dataset_fingerprint"
                ],
                "expected_file_sha256": sha256_file(clean_output_path),
                "expected_replacement_rows": 0,
                "expected_replacement_characters": 0,
                "allow_replacement_character": False,
            },
        },
        "interpretation_rule": (
            "Report the main analysis on all structurally valid official rows and "
            "the paired clean sensitivity separately; never choose between them "
            "after observing scores."
        ),
    }
    _write_json_atomic(policy_output_path, policy)
    return policy


def materialize_clean_sensitivity(
    *,
    input_path: str,
    clean_output_path: str,
    policy_path: str,
) -> Dict[str, Any]:
    """Regenerate only the ignored clean artifact from an existing policy.

    The tracked policy and replacement-row manifest are inputs in this mode,
    never outputs.  This is the safe command collaborators run after clone.
    """

    policy_sha256 = sha256_file(policy_path)
    policy = load_frozen_policy(policy_path)
    base_cfg = {
        "experiment": {
            "status": "validation_pilot_only",
            "dataset": "multi_news",
        },
        "data_policy": {
            "policy_path": policy_path,
            "policy_sha256": policy_sha256,
            "analysis": "main",
        },
    }
    main_preflight = validate_dataset_policy_request(
        base_cfg, input_path, "validation"
    )

    replacement_manifest_path = policy["replacement_character_manifest"]["path"]
    manifest_rows = list(read_jsonl(replacement_manifest_path))
    excluded_ids = [row["id"] for row in manifest_rows]
    if len(excluded_ids) != len(set(excluded_ids)):
        raise ValueError("replacement-character manifest contains duplicate row IDs")
    excluded_id_set = set(excluded_ids)
    observed_excluded_ids: list[str] = []

    def clean_rows() -> Iterable[Dict[str, Any]]:
        for row in read_jsonl(input_path):
            if row["id"] in excluded_id_set:
                observed_excluded_ids.append(row["id"])
                continue
            yield row

    write_jsonl_atomic(clean_output_path, clean_rows())
    if set(observed_excluded_ids) != excluded_id_set or len(
        observed_excluded_ids
    ) != len(excluded_ids):
        missing = sorted(excluded_id_set - set(observed_excluded_ids))
        raise ValueError(
            "main artifact does not contain every frozen sensitivity exclusion "
            f"exactly once; missing={missing}"
        )

    clean_cfg = {
        **base_cfg,
        "data_policy": {
            **base_cfg["data_policy"],
            "analysis": "clean_sensitivity",
        },
    }
    clean_preflight = validate_dataset_policy_request(
        clean_cfg, clean_output_path, "validation"
    )
    return {
        "policy_id": policy["policy_id"],
        "policy_file_sha256": policy_sha256,
        "main": main_preflight,
        "clean_sensitivity": clean_preflight,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/processed/multi_news_validation_canonical.jsonl",
    )
    parser.add_argument(
        "--clean_out",
        default="data/processed/multi_news_validation_clean_sensitivity.jsonl",
    )
    parser.add_argument(
        "--replacement_manifest_out",
        default=(
            "configs/data_policies/"
            "multinews_validation_replacement_rows_v1.jsonl"
        ),
    )
    parser.add_argument(
        "--policy_out",
        default="configs/data_policies/multinews_validation_v1.json",
    )
    parser.add_argument(
        "--exclusion_manifest",
        default=(
            "data/processed/"
            "multi_news_validation_canonical_excluded_rows_manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--initialize_policy",
        action="store_true",
        help=(
            "create/replace the tracked policy and row manifest; use only for a "
            "new version before observing results"
        ),
    )
    args = parser.parse_args()

    if args.initialize_policy:
        result = freeze_policy(
            input_path=args.input,
            clean_output_path=args.clean_out,
            replacement_manifest_path=args.replacement_manifest_out,
            policy_output_path=args.policy_out,
            exclusion_manifest_path=args.exclusion_manifest,
        )
    else:
        result = materialize_clean_sensitivity(
            input_path=args.input,
            clean_output_path=args.clean_out,
            policy_path=args.policy_out,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
