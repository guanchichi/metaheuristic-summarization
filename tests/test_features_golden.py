"""Hand-calculated golden tests for the sentence feature functions.

Every expected value below was derived from the formula by hand, not recorded
from a program run. A snapshot of the current output would have blessed the
existing behaviour, including the length bias documented as F-13(b) in
docs/research/CODE_AUDIT_IEEE_Access.md. The arithmetic is written out so the
next reader can re-derive it instead of having to trust the constant.

If one of these fails, the feature definition changed. Decide whether that was
intended before updating the constant.
"""

import math

import pytest

from src.features.length import length_scores
from src.features.position import position_scores, position_scores_v2
from src.features.tf_isf import sentence_tf_isf_scores, sentence_tf_isf_scores_v2


# Shared four-sentence corpus. Chosen so that sentence frequencies differ and
# one term lands exactly on the zero point of the ISF formula.
#   sentence frequencies over N = 4:  alpha = 3, beta = 2, gamma = 1, delta = 1
CORPUS = ["alpha beta beta", "alpha gamma", "delta", "alpha beta"]


class TestTfIsfGolden:
    """v1 uses legacy ``log(N / (1 + sf))``; v2 uses the corrected,
    non-negative ``log((N + 1) / (sf + 1))`` smoothing."""

    def test_v1_sums_raw_counts_and_divides_by_max(self):
        """v1 has no length normalisation, so repetition alone raises a score.

        ISF: alpha log(4/4)=0, beta log(4/3)=+0.287682,
             gamma log(4/2)=+0.693147, delta log(4/2)=+0.693147

        raw = sum over tokens of count * ISF
          s0 'alpha beta beta' : 1*0 + 2*0.287682 = 0.575364
          s1 'alpha gamma'     : 1*0 + 1*0.693147 = 0.693147
          s2 'delta'           :       1*0.693147 = 0.693147
          s3 'alpha beta'      : 1*0 + 1*0.287682 = 0.287682

        divided by max (0.693147):
          [0.830075, 1.0, 1.0, 0.415037]
        """
        assert sentence_tf_isf_scores(CORPUS) == pytest.approx(
            [0.830075, 1.0, 1.0, 0.415037], abs=1e-6
        )

    def test_v1_rewards_repetition_not_information(self):
        """s0 and s3 carry the same distinct terms; s0 only scores higher
        because 'beta' occurs twice. This is the length/repetition bias that a
        recorded snapshot would have frozen in place."""
        scores = sentence_tf_isf_scores(CORPUS)
        assert scores[0] > scores[3]

    def test_v2_sublinear_tf_and_sqrt_length_normalisation(self):
        """v2 uses smoothed ISF, (1 + log count), and sqrt length.

        ISF with N=4: alpha=log(5/4)=0.223144,
        beta=log(5/3)=0.510826, gamma=delta=log(5/2)=0.916291.

        s0 'alpha beta beta': [0.223144 + (1+log2)*0.510826] / sqrt(3)
                              = 0.628184
        s1 'alpha gamma'    : [0.223144 + 0.916291] / sqrt(2) = 0.805702
        s2 'delta'          : 0.916291
        s3 'alpha beta'     : [0.223144 + 0.510826] / sqrt(2) = 0.518995

        min-max -> [0.274831, 0.721646, 1.0, 0.0]
        """
        assert sentence_tf_isf_scores_v2(CORPUS) == pytest.approx(
            [0.274831, 0.721646, 1.0, 0.0], abs=1e-6
        )

    def test_isf_is_exactly_zero_at_the_break_even_frequency(self):
        """'alpha' appears in 3 of the 4 CORPUS sentences, so
        ISF = log(4 / (1 + 3)) = log(1) = 0 exactly, and it contributes nothing
        to any sentence. That is why s1 and s2 above are driven entirely by
        'gamma' and 'delta'."""
        assert math.log(len(CORPUS) / (1 + 3)) == 0.0

    def test_v2_ubiquitous_term_is_zero_not_negative(self):
        """With corrected smoothing, a term in all four sentences has
        ISF = log((4+1)/(4+1)) = 0 rather than legacy log(4/5) < 0.

        corpus ['alpha', 'alpha rare', 'alpha', 'alpha']; N = 4
        sf: alpha = 4 -> log(5/5) = 0
            rare  = 1 -> log(5/2) = 0.916291

        s0, s2, s3      : 0
        s1 'alpha rare' : 0.916291 / sqrt(2) = 0.647912

        min-max -> [0.0, 1.0, 0.0, 0.0]
        """
        assert math.log((4 + 1) / (4 + 1)) == 0.0
        scores = sentence_tf_isf_scores_v2(["alpha", "alpha rare", "alpha", "alpha"])
        assert scores == pytest.approx([0.0, 1.0, 0.0, 0.0])

    def test_v2_constant_scores_collapse_to_one_half(self):
        """When every sentence scores the same, min-max is undefined and the
        implementation returns 0.5 rather than 0.0 or 1.0."""
        assert sentence_tf_isf_scores_v2(["same word", "same word"]) == pytest.approx(
            [0.5, 0.5]
        )

    def test_empty_corpus(self):
        assert sentence_tf_isf_scores([]) == []
        assert sentence_tf_isf_scores_v2([]) == []


class TestLengthGolden:
    """min(word_count, 40) divided by the largest clipped count *observed in
    this document* -- not by the constant 40. The paper's formula is
    min(len/40, 1), which is a different function; see F-13b."""

    def test_normalises_by_observed_maximum(self):
        """word counts [2, 4, 1]; clip has no effect; max = 4
        -> [2/4, 4/4, 1/4] = [0.5, 1.0, 0.25]"""
        assert length_scores(["a b", "a b c d", "a"]) == pytest.approx(
            [0.5, 1.0, 0.25]
        )

    def test_clip_at_forty_makes_long_sentences_indistinguishable(self):
        """50 and 40 words both clip to 40, so both reach 1.0 and the 50-word
        sentence gains nothing. 20 words -> 20/40 = 0.5."""
        corpus = [" ".join(["w"] * 50), " ".join(["w"] * 40), " ".join(["w"] * 20)]
        assert length_scores(corpus) == pytest.approx([1.0, 1.0, 0.5])

    def test_shorter_document_inflates_scores(self):
        """The same 20-word sentence scores 0.5 above but 1.0 here, because
        normalisation depends on the document. This is why the value is not
        comparable across documents."""
        corpus = [" ".join(["w"] * 20), " ".join(["w"] * 10)]
        assert length_scores(corpus) == pytest.approx([1.0, 0.5])

    def test_empty_corpus(self):
        assert length_scores([]) == []


class TestPositionGolden:
    """All variants are monotonically decreasing in sentence index, i.e. a
    built-in lead prior. The diagnosis in STRATEGY_ASSESSMENT section 1.3
    identified this as the root cause of the pool's lead bias."""

    FOUR = ["s0", "s1", "s2", "s3"]

    def test_v1_linear_decay(self):
        """1 - i/(n-1) with n = 4 -> [1, 2/3, 1/3, 0]"""
        assert position_scores(self.FOUR) == pytest.approx(
            [1.0, 2 / 3, 1 / 3, 0.0]
        )

    def test_v1_last_sentence_scores_exactly_zero(self):
        """The final sentence can never contribute through this feature."""
        assert position_scores(self.FOUR)[-1] == 0.0

    def test_v2_inverse(self):
        """1/(1+i) then divided by the max (which is 1.0 at i = 0)
        -> [1, 1/2, 1/3, 1/4]"""
        assert position_scores_v2(self.FOUR, method="inverse") == pytest.approx(
            [1.0, 0.5, 1 / 3, 0.25]
        )

    def test_v2_exponential(self):
        """exp(-decay * i) with decay = 0.5, divided by max = exp(0) = 1
        -> [1, exp(-0.5), exp(-1), exp(-1.5)]
        =  [1, 0.606531, 0.367879, 0.223130]"""
        assert position_scores_v2(
            self.FOUR, method="exponential", decay=0.5
        ) == pytest.approx([1.0, 0.606531, 0.367879, 0.223130], abs=1e-6)

    def test_v2_inverse_decays_faster_than_linear_early_on(self):
        """Documents the practical difference: 'inverse' halves by the second
        sentence, 'linear' only reaches 2/3."""
        inverse = position_scores_v2(self.FOUR, method="inverse")
        linear = position_scores_v2(self.FOUR, method="linear")
        assert inverse[1] < linear[1]

    def test_single_sentence(self):
        assert position_scores(["only"]) == pytest.approx([1.0])
        assert position_scores_v2(["only"], method="inverse") == pytest.approx([1.0])

    def test_empty_corpus(self):
        assert position_scores([]) == []
        assert position_scores_v2([]) == []
