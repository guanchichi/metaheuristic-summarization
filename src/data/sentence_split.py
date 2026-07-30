"""Shared sentence-boundary tokenizer.

A single ``PunktSentenceTokenizer`` instance, built without any downloaded
NLTK model, is the sentence-boundary authority for both the Multi-News
canonical preprocessing pipeline (``src/data/preprocess_multinews.py``) and
the ROUGE-Lsum evaluator (``src/eval/rouge.py``). They must agree on where
sentences start and end -- computing candidate/selection features on one
segmentation and scoring against another silently changes what "a sentence"
means between the two.

Deliberately not ``nltk.sent_tokenize``/``split_summaries=True``: that path
loads a pretrained model from the ``punkt_tab`` NLTK data package, which is
not vendored or pre-cached anywhere in this repo and raises ``LookupError``
on a machine without prior ``nltk.download``. This tokenizer instead builds
an untrained ``PunktSentenceTokenizer`` seeded only with a manually curated
abbreviation list, so it needs no external resource.
"""

from __future__ import annotations

from typing import List, Tuple

from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize.punkt import PunktParameters


ABBREVIATIONS = frozenset(
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


def build_sentence_tokenizer() -> PunktSentenceTokenizer:
    """Create a deterministic Punkt tokenizer without runtime downloads."""

    parameters = PunktParameters()
    parameters.abbrev_types.update(ABBREVIATIONS)
    return PunktSentenceTokenizer(parameters)


_SENTENCE_TOKENIZER = build_sentence_tokenizer()


def span_tokenize(text: str) -> List[Tuple[int, int]]:
    """Character-offset sentence spans, for callers that need source spans."""

    return list(_SENTENCE_TOKENIZER.span_tokenize(text))


def split_sentences(text: str) -> List[str]:
    """Return *text* split into sentences, using the shared tokenizer."""

    return _SENTENCE_TOKENIZER.tokenize(text)
