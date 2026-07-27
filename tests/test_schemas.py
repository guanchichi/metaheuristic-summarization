"""Tests for the canonical DocumentExample schema contract."""

import pytest

from src.data.schemas import SchemaValidationError, build_document_example


def _one_document_example(*, metadata=None):
    return build_document_example(
        example_id="cluster",
        split="validation",
        documents=[["Only source sentence."]],
        references=["Reference."],
        input_mode="multi_document",
        output_mode="multi_sentence",
        metadata=metadata,
    )


def test_multi_document_single_source_cluster_passes_with_explicit_marker():
    example = _one_document_example(metadata={"n_source_documents": 1})
    assert len(example["documents"]) == 1
    assert example["task_profile"]["input_mode"] == "multi_document"


def test_multi_document_single_source_cluster_fails_without_explicit_marker():
    with pytest.raises(SchemaValidationError, match="n_source_documents"):
        _one_document_example(metadata=None)
    with pytest.raises(SchemaValidationError, match="n_source_documents"):
        _one_document_example(metadata={"n_source_documents": 2})
