"""Reproducible, boundary-preserving preprocessing for Multi-News."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import unicodedata
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize.punkt import PunktParameters
from tqdm import tqdm

from src.data.schemas import build_document_example
from src.utils.io import ensure_dir, write_jsonl_atomic


DATASET_ID = "alexfabbri/multi_news"
DATASET_REVISION = "1f20a01dbf6463236108a8d7fd39f3ae9750dcc3"
DATASET_DATA_DIR = "default"
DATASET_FILES = {
    "train": (
        "multi_news-train-00000-of-00002.parquet",
        "multi_news-train-00001-of-00002.parquet",
    ),
    "validation": ("multi_news-validation.parquet",),
    "test": ("multi_news-test.parquet",),
}
PREPROCESSOR_VERSION = "multinews-canonical-v1"
DOCUMENT_SEPARATOR = "|||||"
_SEPARATOR_RE = re.compile(r"\s*\|\|\|\|\|\s*")
_LEADING_SUMMARY_MARKER_RE = re.compile(r"^\s*[-–—]\s+")


class MultiNewsPreprocessingError(ValueError):
    """Raised when an input row cannot be transformed without guessing."""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def _build_sentence_tokenizer() -> PunktSentenceTokenizer:
    """Create a deterministic Punkt tokenizer without runtime downloads."""

    parameters = PunktParameters()
    parameters.abbrev_types.update(
        {
            "adm",
            "capt",
            "col",
            "dr",
            "e.g",
            "gen",
            "gov",
            "i.e",
            "jr",
            "lt",
            "mr",
            "mrs",
            "ms",
            "prof",
            "rep",
            "rev",
            "sen",
            "sr",
            "st",
            "u.s",
            "vs",
        }
    )
    return PunktSentenceTokenizer(parameters)


_SENTENCE_TOKENIZER = _build_sentence_tokenizer()


def resolve_data_files(
    dataset_id: str,
    revision: str,
    data_dir: str,
    split: str,
) -> List[str]:
    """Resolve the pinned author-hosted parquet files without remote code."""

    if split not in DATASET_FILES:
        raise ValueError(f"unsupported split: {split}")
    root = f"https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/{data_dir}"
    return [f"{root}/{filename}" for filename in DATASET_FILES[split]]


def split_source_documents(raw_cluster: str) -> List[Dict[str, Any]]:
    """Split on the official delimiter while preserving raw character spans."""

    if not isinstance(raw_cluster, str) or not raw_cluster.strip():
        raise MultiNewsPreprocessingError("document field must be a non-empty string")

    documents: List[Dict[str, Any]] = []
    cursor = 0
    matches = list(_SEPARATOR_RE.finditer(raw_cluster))
    boundaries = [(match.start(), match.end()) for match in matches]
    boundaries.append((len(raw_cluster), len(raw_cluster)))
    for boundary_start, boundary_end in boundaries:
        raw_slice = raw_cluster[cursor:boundary_start]
        leading = len(raw_slice) - len(raw_slice.lstrip())
        trailing = len(raw_slice.rstrip())
        text = raw_slice.strip()
        if not text:
            raise MultiNewsPreprocessingError(
                f"empty source document around raw character offset {cursor}"
            )
        start = cursor + leading
        end = cursor + trailing
        documents.append({"text": text, "source_char_start": start, "source_char_end": end})
        cursor = boundary_end

    if len(documents) < 2:
        raise MultiNewsPreprocessingError(
            f"expected at least two source documents separated by {DOCUMENT_SEPARATOR!r}"
        )
    return documents


def segment_document(
    raw_document: str,
    *,
    min_words: int = 1,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Segment one source document and retain cleaned-to-raw span mapping."""

    if min_words < 1:
        raise ValueError("min_words must be at least 1")
    sentences: List[str] = []
    mapping: List[Dict[str, Any]] = []
    for original_position, (start, end) in enumerate(
        _SENTENCE_TOKENIZER.span_tokenize(raw_document)
    ):
        raw_sentence = raw_document[start:end]
        cleaned = _normalize_text(raw_sentence)
        if not cleaned or len(cleaned.split()) < min_words:
            continue
        sentences.append(cleaned)
        mapping.append(
            {
                "original_sentence_position": original_position,
                "document_char_start": start,
                "document_char_end": end,
                "raw_sentence_sha256": _sha256(raw_sentence),
                "whitespace_normalized": cleaned != raw_sentence.strip(),
            }
        )
    if not sentences:
        raise MultiNewsPreprocessingError("source document produced no sentences")
    return sentences, mapping


def process_example(
    example: Mapping[str, Any],
    *,
    split: str,
    row_index: int,
    dataset_id: str = DATASET_ID,
    dataset_revision: str = DATASET_REVISION,
    dataset_data_dir: str = DATASET_DATA_DIR,
    min_words: int = 1,
    is_debug_subset: bool = False,
    allow_replacement_character: bool = False,
) -> Dict[str, Any]:
    """Convert one official Multi-News row to a canonical DocumentExample."""

    raw_cluster = example.get("document")
    raw_summary = example.get("summary")
    if not isinstance(raw_cluster, str) or not isinstance(raw_summary, str):
        raise MultiNewsPreprocessingError("row must contain string 'document' and 'summary' fields")
    if not allow_replacement_character and "\ufffd" in raw_cluster + raw_summary:
        raise MultiNewsPreprocessingError(
            "row contains Unicode replacement character U+FFFD; source must be repaired or "
            "the explicit diagnostic override must be used"
        )

    source_documents = split_source_documents(raw_cluster)
    document_sentences: List[Sequence[str]] = []
    sentence_metadata: List[Sequence[Mapping[str, Any]]] = []
    document_metadata: List[Mapping[str, Any]] = []
    for source_order, source_document in enumerate(source_documents):
        sentences, mapping = segment_document(source_document["text"], min_words=min_words)
        document_sentences.append(sentences)
        sentence_metadata.append(mapping)
        document_metadata.append(
            {
                "original_document_position": source_order,
                "source_char_start": source_document["source_char_start"],
                "source_char_end": source_document["source_char_end"],
                "raw_document_sha256": _sha256(source_document["text"]),
            }
        )

    summary_without_marker = _LEADING_SUMMARY_MARKER_RE.sub("", raw_summary, count=1)
    reference = _normalize_text(summary_without_marker)
    if not reference:
        raise MultiNewsPreprocessingError("summary becomes empty after normalization")
    example_id = example.get("id") or f"{split}_{row_index}"

    return build_document_example(
        example_id=str(example_id),
        split=split,
        documents=document_sentences,
        references=[reference],
        input_mode="multi_document",
        output_mode="multi_sentence",
        dataset_name="Multi-News",
        document_metadata=document_metadata,
        sentence_metadata=sentence_metadata,
        metadata={
            "dataset_id": dataset_id,
            "dataset_revision": dataset_revision,
            "dataset_data_dir": dataset_data_dir,
            "source_row_index": row_index,
            "raw_cluster_sha256": _sha256(raw_cluster),
            "preprocessor_version": PREPROCESSOR_VERSION,
            "document_separator": DOCUMENT_SEPARATOR,
            "min_words": min_words,
            "is_debug_subset": is_debug_subset,
            "replacement_character_override": allow_replacement_character,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=("train", "validation", "test"))
    parser.add_argument("--out", default=None, help="output canonical JSONL path")
    parser.add_argument("--dataset_id", default=DATASET_ID)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--data_dir", default=DATASET_DATA_DIR)
    parser.add_argument("--min_words", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="debug subset only")
    parser.add_argument("--allow_replacement_character", action="store_true")
    args = parser.parse_args()

    if args.min_words != 1 and args.limit is None:
        raise ValueError(
            "--min_words other than 1 filters source content and is allowed only with "
            "--limit for diagnostic subsets"
        )

    data_files = resolve_data_files(
        args.dataset_id,
        args.revision,
        args.data_dir,
        args.split,
    )
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Downloading Multi-News requires the optional 'datasets' package. "
            "Install the full research requirements before running this command."
        ) from exc
    dataset = load_dataset(
        "parquet",
        data_files={args.split: data_files},
        split=args.split,
    )
    if args.limit is not None:
        if args.limit < 1 or args.limit > len(dataset):
            raise ValueError(f"--limit must be between 1 and {len(dataset)}")
        dataset = dataset.select(range(args.limit))

    suffix = f"_limit{args.limit}" if args.limit is not None else ""
    out_path = args.out or os.path.join(
        "data", "processed", f"multi_news_{args.split}_canonical{suffix}.jsonl"
    )
    ensure_dir(os.path.dirname(out_path) or ".")
    rows = (
        process_example(
            example,
            split=args.split,
            row_index=index,
            dataset_id=args.dataset_id,
            dataset_revision=args.revision,
            dataset_data_dir=args.data_dir,
            min_words=args.min_words,
            is_debug_subset=args.limit is not None,
            allow_replacement_character=args.allow_replacement_character,
        )
        for index, example in tqdm(enumerate(dataset), total=len(dataset), desc="Multi-News")
    )
    write_jsonl_atomic(out_path, rows)
    print(f"Wrote {len(dataset)} canonical rows to {out_path}")


if __name__ == "__main__":
    main()
