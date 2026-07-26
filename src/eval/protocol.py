"""Explicit evaluation-protocol boundary.

Metric names alone are not an evaluation protocol.  Tokenization, reference
selection, sentence handling, and aggregation must be fixed before scores can
be compared.  This module prevents accidental cross-dataset evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

from src.eval.rouge import rouge_scores


MULTISENTENCE_LSUM = "multisentence_lsum"
SCITLDR_OFFICIAL = "scitldr_official"
KNOWN_PROTOCOLS = (MULTISENTENCE_LSUM, SCITLDR_OFFICIAL)


class ProtocolUnavailableError(RuntimeError):
    """Raised when a named protocol cannot yet be reproduced faithfully."""


def evaluate_corpus(
    predictions: Sequence[str],
    references: Sequence[Any],
    protocol: str,
) -> Dict[str, float]:
    """Evaluate only under an explicitly named, auditable protocol."""

    if protocol == MULTISENTENCE_LSUM:
        return rouge_scores(predictions, references)
    if protocol == SCITLDR_OFFICIAL:
        raise ProtocolUnavailableError(
            "scitldr_official is intentionally unavailable until the repository "
            "contains a pinned, conformance-tested wrapper for the official "
            "SciTLDR evaluation implementation. Do not substitute generic "
            "rouge-score results and label them official."
        )
    raise ValueError(f"unknown evaluation protocol {protocol!r}; choose one of {KNOWN_PROTOCOLS}")

