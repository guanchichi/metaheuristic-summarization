"""Hand-calculated golden tests for the ROUGE evaluation protocol.

These pin the four decisions that made the ICT Express numbers wrong or
unstable, so a future change cannot quietly undo them:

1. rougeL and rougeLsum are different metrics. The draft reported rougeL for
   multi-sentence summaries, which computes one LCS over the whole summary and
   under-scores extracts (F-2: 0.2014 vs 0.3857 on Multi-News, under the
   sentence splitter in effect when those numbers were recorded).
2. Prediction and reference must be segmented by the same function. Honouring
   the source's stray newlines on one side only silently lowers the score.
3. Multi-reference data selects one reference by max ROUGE-1 and reports every
   metric from that same reference. Taking an independent maximum per metric is
   a different, more optimistic protocol.
4. Sentence boundaries must mean the same thing on the data side and the
   evaluation side. Both now go through the one shared Punkt tokenizer in
   ``src/data/sentence_split.py`` instead of each having its own splitting
   rule.

Expected values are derived by hand below, not recorded from a run.
"""

import pytest
from rouge_score import rouge_scorer

from src.data.sentence_split import span_tokenize, split_sentences
from src.data.preprocess_multinews import segment_document
from src.eval.rouge import (
    _as_lsum,
    rouge_scores,
    rouge_scores_legacy,
    score_single,
)


# Same six tokens on both sides, but the two sentences appear in opposite order.
# Content is identical; only sentence order differs.
REFERENCE = "alpha beta gamma. delta epsilon zeta."
PREDICTION = "delta epsilon zeta. alpha beta gamma."


class TestRougeLVersusLsum:
    def test_rouge_l_penalises_reordered_sentences(self):
        """rougeL flattens the summary into one token sequence:

        reference tokens: alpha beta gamma delta epsilon zeta
        prediction tokens: delta epsilon zeta alpha beta gamma

        The longest common subsequence is 3 ('alpha beta gamma' or
        'delta epsilon zeta' -- they cannot both be used, because taking one
        forces the other out of order).

        precision = 3/6, recall = 3/6, F1 = 0.5
        """
        assert rouge_scores_legacy([PREDICTION], [REFERENCE])[
            "rougeL"
        ] == pytest.approx(0.5)

    def test_rouge_lsum_matches_sentence_by_sentence(self):
        """rougeLsum uses sentence boundaries for summary-level union-LCS:

        the union finds 'alpha beta gamma' intact in one prediction sentence
        and 'delta epsilon zeta' intact in the other -> 6 total hits.

        total 6 of 6 reference tokens, 6 of 6 prediction tokens, F1 = 1.0
        """
        assert rouge_scores([PREDICTION], [REFERENCE])[
            "rougeLsum"
        ] == pytest.approx(1.0)

    def test_the_two_metrics_disagree_by_a_wide_margin(self):
        """0.5 versus 1.0 on identical content. Reporting one while citing
        literature that used the other is not a comparison."""
        legacy = rouge_scores_legacy([PREDICTION], [REFERENCE])["rougeL"]
        current = rouge_scores([PREDICTION], [REFERENCE])["rougeLsum"]
        assert current - legacy == pytest.approx(0.5)

    def test_unigram_and_bigram_scores_are_unaffected_by_segmentation(self):
        """R-1 and R-2 do not depend on sentence boundaries, which is why the
        F-2 correction moved only the ROUGE-L column."""
        scores = rouge_scores([PREDICTION], [REFERENCE])
        assert scores["rouge1"] == pytest.approx(1.0)
        # Reordering breaks the bigrams that span the sentence boundary.
        assert scores["rouge2"] == pytest.approx(0.8)


class TestSegmentationSymmetry:
    """In this project's Multi-News predictions, 370 of 500 carried stray
    newlines inherited from source headlines and image captions, while no
    reference did. Treating those as sentence boundaries segments the two sides
    differently."""

    NOISY_PREDICTION = "delta epsilon\n zeta. alpha beta\n gamma."

    def test_honouring_stray_newlines_understates_the_score(self):
        scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
        asymmetric = scorer.score(
            _as_lsum(REFERENCE, presegmented=True),
            _as_lsum(self.NOISY_PREDICTION, presegmented=True),
        )["rougeLsum"].fmeasure
        symmetric = scorer.score(
            _as_lsum(REFERENCE), _as_lsum(self.NOISY_PREDICTION)
        )["rougeLsum"].fmeasure
        assert asymmetric == pytest.approx(5 / 6)
        assert symmetric == pytest.approx(1.0)
        assert symmetric > asymmetric

    def test_default_normalises_whitespace_before_splitting(self):
        """The default path collapses newlines to spaces, then splits with
        the shared Punkt tokenizer, so both sides are segmented identically."""
        assert _as_lsum(self.NOISY_PREDICTION) == (
            "delta epsilon zeta.\nalpha beta gamma."
        )
        assert _as_lsum(REFERENCE) == "alpha beta gamma.\ndelta epsilon zeta."

    def test_public_api_uses_the_symmetric_path(self):
        assert rouge_scores([self.NOISY_PREDICTION], [REFERENCE])[
            "rougeLsum"
        ] == pytest.approx(1.0)


class TestSharedTokenizerParity:
    """The evaluator (``src/eval/rouge.py``) and the Multi-News canonical
    preprocessing pipeline (``src/data/preprocess_multinews.py``) must agree
    on where a sentence starts and ends. Before this change they used two
    independent splitters (a hand-written regex in the evaluator vs. a Punkt
    tokenizer in preprocessing) with no test tying them together."""

    TEXT = (
        "Mr. Smith met U.S. officials on Tuesday. "
        "Reports said 3.5 million people attended the rally."
    )

    def test_evaluator_and_data_layer_agree_on_sentence_boundaries(self):
        via_data_layer_spans = [
            self.TEXT[start:end] for start, end in span_tokenize(self.TEXT)
        ]
        via_data_layer_segment_document, _ = segment_document(self.TEXT)
        via_evaluator = _as_lsum(self.TEXT).split("\n")

        assert via_data_layer_spans == split_sentences(self.TEXT)
        assert via_evaluator == via_data_layer_spans
        # segment_document additionally whitespace-normalises each sentence,
        # so compare after the same normalisation instead of byte-for-byte.
        assert via_data_layer_segment_document == [
            " ".join(s.split()) for s in via_data_layer_spans
        ]

    def test_abbreviations_are_not_mis_split(self):
        """A naive ``.!?`` regex (the evaluator's old splitter) cuts after
        every abbreviation period. The shared Punkt tokenizer, seeded with
        the abbreviation list in ``src/data/sentence_split.py``, does not."""
        assert split_sentences(self.TEXT) == [
            "Mr. Smith met U.S. officials on Tuesday.",
            "Reports said 3.5 million people attended the rally.",
        ]

        import re

        old_regex_splitter = re.compile(r"(?<=[.!?。！？])\s+")
        old_behaviour = [
            p.strip() for p in old_regex_splitter.split(self.TEXT) if p.strip()
        ]
        assert old_behaviour == [
            "Mr.",
            "Smith met U.S.",
            "officials on Tuesday.",
            "Reports said 3.5 million people attended the rally.",
        ]
        assert old_behaviour != split_sentences(self.TEXT)


class TestMultiReferenceSelection:
    """SciTLDR's official script picks the reference with the highest ROUGE-1
    and reports R-1/R-2/R-L from that one reference (allenai/scitldr,
    scripts/cal-rouge.py, ``_get_rouge``). These tests pin that selection rule
    only; they do not establish numerical parity between local ``rouge-score``
    and the official ``files2rouge`` backend."""

    PREDICTION = "aa bb cc dd ee"
    # Every unigram matches, but no bigram does: the reference reorders them.
    REF_HIGH_UNIGRAM = "aa cc ee bb dd"
    # Fewer unigrams match, but 'aa bb' and 'bb cc' survive.
    REF_HIGH_BIGRAM = "aa bb cc xx yy"

    def test_each_reference_alone(self):
        """R-1 = 1.0 / R-2 = 0.0 for the reordered reference;
        R-1 = 0.6 / R-2 = 0.5 for the truncated one."""
        high_unigram = score_single(self.PREDICTION, [self.REF_HIGH_UNIGRAM])
        high_bigram = score_single(self.PREDICTION, [self.REF_HIGH_BIGRAM])
        assert high_unigram["rouge1"] == pytest.approx(1.0)
        assert high_unigram["rouge2"] == pytest.approx(0.0)
        assert high_bigram["rouge1"] == pytest.approx(0.6)
        assert high_bigram["rouge2"] == pytest.approx(0.5)

    def test_all_metrics_come_from_the_reference_chosen_by_rouge1(self):
        """The reordered reference wins on ROUGE-1, so ROUGE-2 must be reported
        as 0.0 from that same reference -- not 0.5 from the other one."""
        scores = score_single(
            self.PREDICTION, [self.REF_HIGH_UNIGRAM, self.REF_HIGH_BIGRAM]
        )
        assert scores["rouge1"] == pytest.approx(1.0)
        assert scores["rouge2"] == pytest.approx(0.0)

    def test_independent_per_metric_maximum_would_be_more_optimistic(self):
        """Guards against someone 'fixing' the selection rule into a per-metric
        maximum, which would report 0.5 for ROUGE-2 here."""
        combined = score_single(
            self.PREDICTION, [self.REF_HIGH_UNIGRAM, self.REF_HIGH_BIGRAM]
        )
        per_metric_max = max(
            score_single(self.PREDICTION, [self.REF_HIGH_UNIGRAM])["rouge2"],
            score_single(self.PREDICTION, [self.REF_HIGH_BIGRAM])["rouge2"],
        )
        assert per_metric_max == pytest.approx(0.5)
        assert combined["rouge2"] < per_metric_max

    def test_reference_selection_metric_is_configurable_but_defaults_to_rouge1(self):
        """Selecting by ROUGE-2 instead picks the other reference, which shows
        the choice is load-bearing and must be stated in the paper."""
        by_bigram = score_single(
            self.PREDICTION,
            [self.REF_HIGH_UNIGRAM, self.REF_HIGH_BIGRAM],
            reference_metric="rouge2",
        )
        assert by_bigram["rouge2"] == pytest.approx(0.5)
        assert by_bigram["rouge1"] == pytest.approx(0.6)


class TestCorpusGuards:
    def test_length_mismatch_fails(self):
        with pytest.raises(ValueError):
            rouge_scores(["a"], ["x", "y"])

    def test_empty_corpus_fails(self):
        with pytest.raises(ValueError):
            rouge_scores([], [])

    def test_empty_reference_list_fails(self):
        with pytest.raises(ValueError):
            score_single("a", [])

    def test_per_example_scores_align_with_the_mean(self):
        """Needed for the paired bootstrap that Gate 3 requires."""
        preds = [PREDICTION, "alpha beta gamma."]
        refs = [REFERENCE, REFERENCE]
        means, per_example = rouge_scores(preds, refs, return_per_example=True)
        assert len(per_example) == 2
        recomputed = sum(row["rougeLsum"] for row in per_example) / 2
        assert means["rougeLsum"] == pytest.approx(recomputed)
