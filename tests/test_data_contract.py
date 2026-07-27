"""Tests for canonical dataset construction and validation."""

import copy

import pytest

from src.data.preprocess_scitldr import process_example
from src.data.schemas import (
    SchemaValidationError,
    build_document_example,
    extract_references,
    flatten_sentence_texts,
    validate_document_example,
)
from src.data.validate_dataset import validate_jsonl


def canonical_example():
    return build_document_example(
        example_id="paper-1",
        split="test",
        documents=[["First sentence.", "Second sentence."]],
        references=["Reference A.", "Reference B."],
        input_mode="single_document",
        output_mode="single_sentence",
        dataset_name="fixture",
    )


def test_canonical_example_round_trip():
    example = canonical_example()
    validate_document_example(example)
    assert flatten_sentence_texts(example) == ["First sentence.", "Second sentence."]
    assert extract_references(example) == ["Reference A.", "Reference B."]
    assert len(example["data_fingerprint"]) == 64


def test_fingerprint_detects_content_mutation():
    example = copy.deepcopy(canonical_example())
    example["documents"][0]["sections"][0]["sentences"][0]["text"] = "Tampered."
    with pytest.raises(SchemaValidationError, match="fingerprint"):
        validate_document_example(example)


def test_builder_rejects_misaligned_provenance_metadata():
    with pytest.raises(SchemaValidationError, match="sentence_metadata"):
        build_document_example(
            example_id="bad-mapping",
            split="test",
            documents=[["First.", "Second."]],
            references=["Reference."],
            input_mode="single_document",
            output_mode="multi_sentence",
            sentence_metadata=[[{"original_sentence_position": 0}]],
        )


def test_scitldr_alternative_references_are_not_concatenated():
    row = process_example(
        {
            "paper_id": "p1",
            "source": ["Sentence one.", "Sentence two."],
            "target": ["Alternative one.", "Alternative two."],
            "source_labels": [0],
            "rouge_scores": [0.5, 0.2],
        },
        split="validation",
    )
    assert row["references"] == ["Alternative one.", "Alternative two."]
    assert "highlights" not in row
    assert row["task_profile"]["output_mode"] == "single_sentence"


def test_dataset_validator_rejects_duplicate_ids(monkeypatch):
    example = canonical_example()
    monkeypatch.setattr(
        "src.data.validate_dataset.read_jsonl",
        lambda _path: iter([example, example]),
    )
    report = validate_jsonl("fixture.jsonl", expected_split="test")
    assert report["valid"] is False
    assert len(report["dataset_fingerprint"]) == 64
    assert "duplicate example id" in report["errors"][0]["error"]


def test_dataset_health_report_contains_streaming_statistics(monkeypatch):
    example = canonical_example()
    monkeypatch.setattr(
        "src.data.validate_dataset.read_jsonl",
        lambda _path: iter([example]),
    )
    report = validate_jsonl("fixture.jsonl", expected_split="test", expected_rows=1)
    assert report["valid"] is True
    assert report["health"]["documents_per_example"] == {1: 1}
    assert report["health"]["references_per_example"] == {2: 1}
    assert report["health"]["sentences_per_example"]["under_20"] == 1
    assert report["health"]["sentence_words"]["max"] == 2
    assert report["health"]["unicode_replacement_characters"] == 0


def test_dataset_validator_rejects_debug_subset_by_default(monkeypatch):
    example = build_document_example(
        example_id="debug-row",
        split="validation",
        documents=[["A sentence."]],
        references=["Reference."],
        input_mode="single_document",
        output_mode="multi_sentence",
        metadata={"is_debug_subset": True},
    )
    monkeypatch.setattr(
        "src.data.validate_dataset.read_jsonl",
        lambda _path: iter([example]),
    )
    report = validate_jsonl("debug.jsonl", expected_split="validation")
    assert report["valid"] is False
    assert "debug-subset" in report["errors"][0]["error"]


def test_dataset_validator_enforces_pinned_revision(monkeypatch):
    example = build_document_example(
        example_id="revision-row",
        split="test",
        documents=[["A sentence."]],
        references=["Reference."],
        input_mode="single_document",
        output_mode="multi_sentence",
        metadata={"dataset_revision": "actual-revision"},
    )
    monkeypatch.setattr(
        "src.data.validate_dataset.read_jsonl",
        lambda _path: iter([example]),
    )
    report = validate_jsonl(
        "revision.jsonl",
        expected_dataset_revision="expected-revision",
    )
    assert report["valid"] is False
    assert "actual-revision" in report["errors"][0]["error"]


def test_policy_error_does_not_remove_row_from_health_denominators(monkeypatch):
    example = build_document_example(
        example_id="damaged-row",
        split="validation",
        documents=[["A damaged \ufffd sentence."]],
        references=["Reference."],
        input_mode="single_document",
        output_mode="multi_sentence",
        dataset_name="fixture",
    )
    monkeypatch.setattr(
        "src.data.validate_dataset.read_jsonl",
        lambda _path: iter([example]),
    )

    strict = validate_jsonl("damaged.jsonl", expected_split="validation")
    assert strict["valid"] is False
    assert strict["health"]["structurally_valid_rows"] == 1
    assert strict["health"]["split_counts"] == {"validation": 1}
    assert strict["documents"] == 1
    assert strict["sentences"] == 1
    assert strict["health"]["unicode_replacement_rows"] == 1

    allowed = validate_jsonl(
        "damaged.jsonl",
        expected_split="validation",
        allow_replacement_character=True,
    )
    assert allowed["valid"] is True
    assert allowed["health"]["structurally_valid_rows"] == 1
