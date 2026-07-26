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
