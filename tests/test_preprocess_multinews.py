"""Golden tests for boundary-preserving Multi-News preprocessing."""

import importlib
import sys

import pytest

from src.data.preprocess_multinews import (
    DegenerateRowError,
    MultiNewsPreprocessingError,
    build_canonical_rows,
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
    source_documents, dropped_segments = split_source_documents(raw_example()["document"])
    assert dropped_segments == []
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


def test_single_source_document_row_is_accepted_and_flagged():
    row = process_example(
        {"document": "Only one source document with enough words.", "summary": "A summary."},
        split="test",
        row_index=0,
    )
    assert len(row["documents"]) == 1
    assert row["task_profile"]["input_mode"] == "multi_document"
    assert row["metadata"]["n_source_documents"] == 1
    assert row["metadata"]["dropped_empty_segments"]["count"] == 0


def test_trailing_extra_delimiter_drops_empty_segment_and_records_it():
    documents, dropped = split_source_documents(
        "First document. ||||| Second document. ||||| "
    )
    assert [doc["text"] for doc in documents] == ["First document.", "Second document."]
    assert len(dropped) == 1
    assert dropped[0]["raw_segment_position"] == 2

    row = process_example(
        {
            "document": "First document. ||||| Second document. ||||| ",
            "summary": "A summary.",
        },
        split="test",
        row_index=0,
    )
    assert len(row["documents"]) == 2
    assert row["metadata"]["n_source_documents"] == 2
    assert row["metadata"]["dropped_empty_segments"]["count"] == 1


def test_empty_document_between_delimiters_is_dropped_not_failed():
    documents, dropped = split_source_documents("First document. ||||| ||||| Third document.")
    assert [doc["text"] for doc in documents] == ["First document.", "Third document."]
    assert len(dropped) == 1


def test_wholly_empty_document_raises_degenerate_row_error():
    with pytest.raises(DegenerateRowError):
        split_source_documents("")
    with pytest.raises(DegenerateRowError):
        process_example({"document": "   ", "summary": "A summary."}, split="test", row_index=0)


def test_degenerate_rows_are_excluded_from_output_and_recorded_in_manifest():
    dataset = [
        {"document": "Only one source document with enough words.", "summary": "Kept summary."},
        {"document": "", "summary": "Never written."},
        {"document": "First. ||||| Second.", "summary": "Also kept."},
    ]
    written_rows, excluded_rows = build_canonical_rows(
        dataset, split="test", show_progress=False
    )
    assert len(written_rows) == 2
    assert [row["metadata"]["source_row_index"] for row in written_rows] == [0, 2]
    assert len(excluded_rows) == 1
    assert excluded_rows[0]["source_row_index"] == 1
    assert "empty" in excluded_rows[0]["reason"]


def test_summary_empty_after_normalization_fails_closed():
    with pytest.raises(MultiNewsPreprocessingError, match="summary becomes empty"):
        process_example(
            {"document": "First. ||||| Second.", "summary": "-  "},
            split="test",
            row_index=0,
        )


def test_unicode_replacement_character_in_document_is_flagged_not_rejected():
    example = raw_example()
    example["document"] = example["document"].replace("result", "res\ufffdl\ufffdt")
    row = process_example(example, split="test", row_index=0)
    assert row["metadata"]["contains_replacement_character"] is True
    assert row["metadata"]["replacement_character_count"] == {"document": 2, "summary": 0}


def test_unicode_replacement_character_in_summary_is_flagged_not_rejected():
    example = raw_example()
    example["summary"] = example["summary"].replace("compact", "comp\ufffdct")
    row = process_example(example, split="test", row_index=0)
    assert row["metadata"]["contains_replacement_character"] is True
    assert row["metadata"]["replacement_character_count"] == {"document": 0, "summary": 1}


def test_clean_row_is_not_falsely_flagged_for_replacement_character():
    row = process_example(raw_example(), split="test", row_index=0)
    assert row["metadata"]["contains_replacement_character"] is False
    assert row["metadata"]["replacement_character_count"] == {"document": 0, "summary": 0}
