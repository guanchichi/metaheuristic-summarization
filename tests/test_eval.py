"""Golden tests for the audited evaluation semantics."""

import pytest

from src.eval.rouge import rouge_scores, score_single


def test_multireference_uses_one_reference_selected_by_rouge1():
    # Reference 1 has the higher R1; reference 2 has the higher R2.  SciTLDR's
    # official protocol selects by R1 and must report every metric from that
    # same reference, rather than taking a per-metric optimistic maximum.
    scores = score_single(
        "a b c d",
        ["a c d", "a b c x y z"],
        metrics=("rouge1", "rouge2", "rougeL"),
    )

    assert scores["rouge1"] == pytest.approx(6 / 7)
    assert scores["rouge2"] == pytest.approx(0.4)
    assert scores["rougeL"] == pytest.approx(6 / 7)


def test_prediction_reference_length_mismatch_fails():
    with pytest.raises(ValueError, match="length mismatch"):
        rouge_scores(["prediction"], [])


def test_empty_corpus_fails():
    with pytest.raises(ValueError, match="empty corpus"):
        rouge_scores([], [])
