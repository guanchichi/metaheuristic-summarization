"""Golden tests for boundary-preserving Multi-News preprocessing."""

import importlib
import sys

import pytest

from src.data.preprocess_multinews import (
    MultiNewsPreprocessingError,
    process_example,
    resolve_data_files,
    segment_document,
    split_source_documents,
)
from src.data.schemas import flatten_sentence_texts, validate_document_example


def test_preprocessor_module_import_does_not_require_datasets(monkeypatch):
    module_name = "src.data.preprocess_multinews"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setitem(sys.modules, "datasets", None)
    imported = importlib.import_module(module_name)
    assert imported.DATASET_ID == "alexfabbri/multi_news"


def raw_example():
    return {
        "document": (
            "Dr. Smith reported the result. A second sentence follows. "
            "||||| Other reporters confirmed it.\nTheir report had more detail."
        ),
        "summary": "–  A compact event summary.\nIt has two sentences.",
    }


def test_official_parquet_paths_are_revision_pinned():
    urls = resolve_data_files("alexfabbri/multi_news", "abc123", "default", "train")
    assert len(urls) == 2
    assert all("/resolve/abc123/default/" in url for url in urls)
    assert urls[0].endswith("multi_news-train-00000-of-00002.parquet")


def test_multinews_documents_and_sentence_order_are_preserved():
    row = process_example(raw_example(), split="validation", row_index=7)
    validate_document_example(row)

    assert row["id"] == "validation_7"
    assert len(row["documents"]) == 2
    assert [doc["source_order"] for doc in row["documents"]] == [0, 1]
    assert flatten_sentence_texts(row) == [
        "Dr. Smith reported the result.",
        "A second sentence follows.",
        "Other reporters confirmed it.",
        "Their report had more detail.",
    ]
    assert row["references"] == ["A compact event summary. It has two sentences."]


def test_raw_document_and_sentence_spans_are_traceable():
    source_documents = split_source_documents(raw_example()["document"])
    row = process_example(raw_example(), split="test", row_index=1)

    for source_document, canonical_document in zip(source_documents, row["documents"]):
        assert canonical_document["metadata"]["source_char_start"] == source_document[
            "source_char_start"
        ]
        for sentence in canonical_document["sections"][0]["sentences"]:
            metadata = sentence["metadata"]
            raw_span = source_document["text"][
                metadata["document_char_start"] : metadata["document_char_end"]
            ]
            assert " ".join(raw_span.split()) == sentence["text"]


def test_filtered_sentence_mapping_retains_original_position():
    sentences, mapping = segment_document(
        "Tiny. This sentence is retained.",
        min_words=2,
    )
    assert sentences == ["This sentence is retained."]
    assert mapping[0]["original_sentence_position"] == 1


def test_missing_multidocument_delimiter_fails():
    with pytest.raises(MultiNewsPreprocessingError, match="at least two"):
        process_example(
            {"document": "Only one source document.", "summary": "A summary."},
            split="test",
            row_index=0,
        )


def test_empty_document_between_delimiters_fails():
    with pytest.raises(MultiNewsPreprocessingError, match="empty source document"):
        split_source_documents("First document. ||||| ||||| Third document.")


def test_unicode_replacement_character_fails_closed():
    example = raw_example()
    example["document"] = example["document"].replace("result", "res\ufffdlt")
    with pytest.raises(MultiNewsPreprocessingError, match=r"U\+FFFD"):
        process_example(example, split="test", row_index=0)
