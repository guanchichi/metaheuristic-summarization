"""Canonical data contracts for summarization experiments.

The project historically used several incompatible flat JSONL formats.  This
module is the single boundary between those legacy rows and the canonical
document representation used by new code.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional


SCHEMA_VERSION = "1.0"
VALID_SPLITS = frozenset({"train", "validation", "test"})
VALID_INPUT_MODES = frozenset({"single_document", "multi_document"})
VALID_OUTPUT_MODES = frozenset({"single_sentence", "multi_sentence"})


class SchemaValidationError(ValueError):
    """Raised when a row violates the canonical data contract."""


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{field} must be a non-empty string")
    return value.strip()


def normalize_references(value: Any) -> List[str]:
    """Normalize a reference field without combining alternative summaries."""

    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, (list, tuple)):
        raise SchemaValidationError("references must be a string or a list of strings")
    refs: List[str] = []
    for i, reference in enumerate(value):
        refs.append(_nonempty_string(reference, f"references[{i}]"))
    return refs


def compute_data_fingerprint(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 fingerprint for the content-bearing fields."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_document_example(
    *,
    example_id: str,
    split: str,
    documents: Iterable[Iterable[str]],
    references: Any,
    input_mode: str,
    output_mode: str,
    dataset_name: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct and validate a canonical ``DocumentExample`` dictionary."""

    example_id = _nonempty_string(example_id, "id")
    canonical_documents: List[Dict[str, Any]] = []
    for document_position, sentence_texts in enumerate(documents):
        document_id = f"{example_id}:d{document_position:03d}"
        sentences = []
        for sentence_position, text in enumerate(sentence_texts):
            text = _nonempty_string(
                text,
                f"documents[{document_position}].sentences[{sentence_position}]",
            )
            sentences.append(
                {
                    "sentence_id": f"{document_id}:s{sentence_position:06d}",
                    "text": text,
                    "document_position": sentence_position,
                    "section_position": sentence_position,
                }
            )
        canonical_documents.append(
            {
                "document_id": document_id,
                "source_order": document_position,
                "sections": [
                    {
                        "section_id": f"{document_id}:section:000",
                        "heading": None,
                        "sentences": sentences,
                    }
                ],
            }
        )

    refs = normalize_references(references)
    task_profile = {"input_mode": input_mode, "output_mode": output_mode}
    example: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "id": example_id,
        "split": split,
        "documents": canonical_documents,
        "references": refs,
        "task_profile": task_profile,
    }
    if dataset_name is not None:
        example["dataset_name"] = _nonempty_string(dataset_name, "dataset_name")
    if metadata is not None:
        example["metadata"] = dict(metadata)
    example["data_fingerprint"] = compute_data_fingerprint(example)
    validate_document_example(example)
    return example


def validate_document_example(example: Mapping[str, Any]) -> None:
    """Fail loudly if ``example`` is not a canonical DocumentExample."""

    if example.get("schema_version") != SCHEMA_VERSION:
        raise SchemaValidationError(
            f"schema_version must be {SCHEMA_VERSION!r}; got {example.get('schema_version')!r}"
        )
    _nonempty_string(example.get("id"), "id")

    split = example.get("split")
    if split not in VALID_SPLITS:
        raise SchemaValidationError(f"split must be one of {sorted(VALID_SPLITS)}")

    task_profile = example.get("task_profile")
    if not isinstance(task_profile, Mapping):
        raise SchemaValidationError("task_profile must be an object")
    if task_profile.get("input_mode") not in VALID_INPUT_MODES:
        raise SchemaValidationError(
            f"task_profile.input_mode must be one of {sorted(VALID_INPUT_MODES)}"
        )
    if task_profile.get("output_mode") not in VALID_OUTPUT_MODES:
        raise SchemaValidationError(
            f"task_profile.output_mode must be one of {sorted(VALID_OUTPUT_MODES)}"
        )

    documents = example.get("documents")
    if not isinstance(documents, list) or not documents:
        raise SchemaValidationError("documents must be a non-empty list")
    if task_profile.get("input_mode") == "single_document" and len(documents) != 1:
        raise SchemaValidationError("single_document examples must contain exactly one document")
    if task_profile.get("input_mode") == "multi_document" and len(documents) < 2:
        raise SchemaValidationError("multi_document examples must contain at least two documents")

    document_ids = set()
    section_ids = set()
    sentence_ids = set()
    for doc_i, document in enumerate(documents):
        if not isinstance(document, Mapping):
            raise SchemaValidationError(f"documents[{doc_i}] must be an object")
        document_id = _nonempty_string(document.get("document_id"), f"documents[{doc_i}].document_id")
        if document_id in document_ids:
            raise SchemaValidationError(f"duplicate document_id: {document_id}")
        document_ids.add(document_id)
        if document.get("source_order") != doc_i:
            raise SchemaValidationError(f"documents[{doc_i}].source_order must equal {doc_i}")
        sections = document.get("sections")
        if not isinstance(sections, list) or not sections:
            raise SchemaValidationError(f"documents[{doc_i}].sections must be a non-empty list")
        expected_document_position = 0
        for section_i, section in enumerate(sections):
            if not isinstance(section, Mapping):
                raise SchemaValidationError(
                    f"documents[{doc_i}].sections[{section_i}] must be an object"
                )
            section_id = _nonempty_string(
                section.get("section_id"),
                f"documents[{doc_i}].sections[{section_i}].section_id",
            )
            if section_id in section_ids:
                raise SchemaValidationError(f"duplicate section_id: {section_id}")
            section_ids.add(section_id)
            heading = section.get("heading")
            if heading is not None and not isinstance(heading, str):
                raise SchemaValidationError(
                    f"documents[{doc_i}].sections[{section_i}].heading must be a string or null"
                )
            sentences = section.get("sentences")
            if not isinstance(sentences, list):
                raise SchemaValidationError(
                    f"documents[{doc_i}].sections[{section_i}].sentences must be a list"
                )
            for sentence_i, sentence in enumerate(sentences):
                prefix = f"documents[{doc_i}].sections[{section_i}].sentences[{sentence_i}]"
                if not isinstance(sentence, Mapping):
                    raise SchemaValidationError(f"{prefix} must be an object")
                sentence_id = _nonempty_string(sentence.get("sentence_id"), f"{prefix}.sentence_id")
                if sentence_id in sentence_ids:
                    raise SchemaValidationError(f"duplicate sentence_id: {sentence_id}")
                sentence_ids.add(sentence_id)
                _nonempty_string(sentence.get("text"), f"{prefix}.text")
                if sentence.get("document_position") != expected_document_position:
                    raise SchemaValidationError(
                        f"{prefix}.document_position must equal {expected_document_position}"
                    )
                if sentence.get("section_position") != sentence_i:
                    raise SchemaValidationError(f"{prefix}.section_position must equal {sentence_i}")
                expected_document_position += 1
        if expected_document_position == 0:
            raise SchemaValidationError(f"documents[{doc_i}] must contain at least one sentence")

    normalize_references(example.get("references"))
    fingerprint = example.get("data_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise SchemaValidationError("data_fingerprint must be a 64-character SHA-256 digest")
    expected_fingerprint = compute_data_fingerprint(
        {key: value for key, value in example.items() if key != "data_fingerprint"}
    )
    if fingerprint != expected_fingerprint:
        raise SchemaValidationError("data_fingerprint does not match the example content")


def flatten_sentence_texts(example: Mapping[str, Any]) -> List[str]:
    """Read sentences from canonical rows, while accepting legacy rows at the edge."""

    if "documents" not in example:
        sentences = example.get("sentences", [])
        if not isinstance(sentences, list) or not all(isinstance(s, str) for s in sentences):
            raise SchemaValidationError("legacy sentences must be a list of strings")
        return list(sentences)

    validate_document_example(example)
    return [
        sentence["text"]
        for document in sorted(example["documents"], key=lambda item: item["source_order"])
        for section in document["sections"]
        for sentence in section["sentences"]
    ]


def extract_references(example: Mapping[str, Any]) -> List[str]:
    """Read canonical references or normalize legacy ``highlights``/``reference``."""

    if "references" in example:
        return normalize_references(example["references"])
    if "highlights" in example:
        return normalize_references(example["highlights"])
    return normalize_references(example.get("reference"))
