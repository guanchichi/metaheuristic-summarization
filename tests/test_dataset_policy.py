"""Regression tests for frozen dataset policy generation and enforcement."""

import json
from pathlib import Path
import shutil
import uuid

import pytest

from src.data.freeze_multinews_policy import (
    freeze_policy,
    materialize_clean_sensitivity,
)
from src.data.policy import sha256_file, validate_dataset_policy_request
from src.data.preprocess_multinews import DATASET_REVISION
from src.data.schemas import build_document_example
from src.utils.io import write_jsonl


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def policy_tmp_dir():
    # The managed Windows test host can deny pytest's mode-0700 ``tmp_path``
    # directories.  A unique ignored directory created with normal workspace
    # permissions is equivalent and keeps the regression portable.
    path = ROOT / "data" / "processed" / f"policy_test_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _row(example_id: str, text: str):
    return build_document_example(
        example_id=example_id,
        split="validation",
        documents=[[text]],
        references=["Reference."],
        input_mode="multi_document",
        output_mode="multi_sentence",
        dataset_name="Multi-News",
        metadata={
            "dataset_revision": DATASET_REVISION,
            "source_row_index": int(example_id.rsplit("_", 1)[-1]),
            "n_source_documents": 1,
        },
    )


@pytest.fixture
def frozen_fixture(policy_tmp_dir):
    main_path = policy_tmp_dir / "main.jsonl"
    clean_path = policy_tmp_dir / "clean.jsonl"
    manifest_path = policy_tmp_dir / "replacement.jsonl"
    policy_path = policy_tmp_dir / "policy.json"
    write_jsonl(
        str(main_path),
        [
            _row("validation_0", "Clean source sentence."),
            _row("validation_1", "Damaged \ufffd source sentence."),
            _row("validation_2", "Another clean source sentence."),
        ],
    )
    policy = freeze_policy(
        input_path=str(main_path),
        clean_output_path=str(clean_path),
        replacement_manifest_path=str(manifest_path),
        policy_output_path=str(policy_path),
        exclusion_manifest_path=None,
        expected_main_rows=3,
        expected_replacement_rows=1,
        expected_replacement_characters=1,
    )
    return {
        "main": main_path,
        "clean": clean_path,
        "manifest": manifest_path,
        "policy_path": policy_path,
        "policy": policy,
    }


def _config(policy_path, analysis):
    return {
        "experiment": {
            "status": "validation_pilot_only",
            "dataset": "multi_news",
        },
        "data_policy": {
            "policy_path": str(policy_path),
            "policy_sha256": sha256_file(str(policy_path)),
            "analysis": analysis,
        },
    }


def test_freeze_creates_exact_paired_clean_subset(frozen_fixture):
    policy = frozen_fixture["policy"]
    manifest_rows = [
        json.loads(line)
        for line in frozen_fixture["manifest"].read_text(encoding="utf-8").splitlines()
    ]
    clean_ids = [
        json.loads(line)["id"]
        for line in frozen_fixture["clean"].read_text(encoding="utf-8").splitlines()
    ]

    assert clean_ids == ["validation_0", "validation_2"]
    assert [row["id"] for row in manifest_rows] == ["validation_1"]
    assert manifest_rows[0]["total_replacement_characters"] == 1
    assert policy["analyses"]["main"]["expected_rows"] == 3
    assert policy["analyses"]["clean_sensitivity"]["expected_rows"] == 2
    assert policy["analyses"]["main"]["text_repair"] == "forbidden"


def test_runtime_accepts_only_the_artifact_bound_to_each_analysis(frozen_fixture):
    main_report = validate_dataset_policy_request(
        _config(frozen_fixture["policy_path"], "main"),
        str(frozen_fixture["main"]),
        "validation",
    )
    clean_report = validate_dataset_policy_request(
        _config(frozen_fixture["policy_path"], "clean_sensitivity"),
        str(frozen_fixture["clean"]),
        "validation",
    )

    assert main_report["rows"] == 3
    assert main_report["replacement_character_rows"] == 1
    assert clean_report["rows"] == 2
    assert clean_report["replacement_character_rows"] == 0
    with pytest.raises(ValueError, match="frozen policy"):
        validate_dataset_policy_request(
            _config(frozen_fixture["policy_path"], "main"),
            str(frozen_fixture["clean"]),
            "validation",
        )


def test_runtime_rejects_experiment_policy_dataset_mismatch(frozen_fixture):
    cfg = _config(frozen_fixture["policy_path"], "main")
    cfg["experiment"]["dataset"] = "GovReport"
    with pytest.raises(ValueError, match="does not match frozen policy dataset"):
        validate_dataset_policy_request(
            cfg,
            str(frozen_fixture["main"]),
            "validation",
        )


def test_runtime_rejects_manifest_tampering(frozen_fixture):
    with frozen_fixture["manifest"].open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="manifest SHA-256"):
        validate_dataset_policy_request(
            _config(frozen_fixture["policy_path"], "main"),
            str(frozen_fixture["main"]),
            "validation",
        )


def test_runtime_rejects_policy_tampering(frozen_fixture):
    cfg = _config(frozen_fixture["policy_path"], "main")
    with frozen_fixture["policy_path"].open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="policy SHA-256"):
        validate_dataset_policy_request(
            cfg,
            str(frozen_fixture["main"]),
            "validation",
        )


def test_materialize_recreates_only_ignored_clean_artifact(frozen_fixture):
    policy_before = frozen_fixture["policy_path"].read_bytes()
    manifest_before = frozen_fixture["manifest"].read_bytes()
    frozen_fixture["clean"].unlink()

    report = materialize_clean_sensitivity(
        input_path=str(frozen_fixture["main"]),
        clean_output_path=str(frozen_fixture["clean"]),
        policy_path=str(frozen_fixture["policy_path"]),
    )

    assert report["clean_sensitivity"]["rows"] == 2
    assert frozen_fixture["policy_path"].read_bytes() == policy_before
    assert frozen_fixture["manifest"].read_bytes() == manifest_before


def test_freeze_fails_when_observed_damage_does_not_match_declared_policy(
    policy_tmp_dir,
):
    main_path = policy_tmp_dir / "main.jsonl"
    write_jsonl(str(main_path), [_row("validation_0", "Clean sentence.")])
    with pytest.raises(ValueError, match="policy precheck"):
        freeze_policy(
            input_path=str(main_path),
            clean_output_path=str(policy_tmp_dir / "clean.jsonl"),
            replacement_manifest_path=str(policy_tmp_dir / "manifest.jsonl"),
            policy_output_path=str(policy_tmp_dir / "policy.json"),
            expected_main_rows=1,
            expected_replacement_rows=1,
            expected_replacement_characters=1,
        )
