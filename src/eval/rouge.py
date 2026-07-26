"""ROUGE evaluation.

IMPORTANT (2026-07 audit fix):
    The new internal protocol for multi-sentence summaries uses ROUGE-Lsum.

    ``rouge_score``'s ``rougeL`` treats the whole summary as one token
    sequence and computes a single LCS, which severely under-scores
    multi-sentence extracts.  ``rougeLsum`` splits on newlines, matches
    sentence-by-sentence and aggregates.  Published results produced with
    Perl ROUGE / pyrouge are not automatically comparable; every local
    baseline must be rescored through the same evaluator.

    Reproduced legacy diagnostics on Multi-News (n=5622):
        full_benchmark_result: rougeL=0.2014, rougeLsum=0.3857
        ExpB_K20_Max_Coverage: rougeL=0.2019, rougeLsum=0.3880
    Both runs are test-tuned legacy artifacts and are invalid as new-paper
    results; the numbers only demonstrate metric sensitivity.

    ``rougeLsum`` only works if BOTH prediction and reference have their
    sentences separated by ``\\n``.  ``_as_lsum`` below guarantees that.
"""

from typing import Dict, List, Sequence, Union
import re

# Split after . ! ? (also CJK 。！？) followed by whitespace.
_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")
_WS = re.compile(r"\s+")

DEFAULT_METRICS = ("rouge1", "rouge2", "rougeLsum")


def _as_lsum(text: str, presegmented: bool = False) -> str:
    """Return *text* with one sentence per line, as required by rougeLsum.

    Both prediction and reference MUST be segmented the same way, otherwise
    the LCS matching is asymmetric and the score is meaningless.

    Raw newlines in the input are treated as *whitespace noise*, not sentence
    boundaries -- in this project's Multi-News data 370/500 predictions carry
    stray "\\n" from the source articles (headlines, image captions) while 0
    references do.  Honouring those newlines would segment the two sides
    differently and understate ROUGE-Lsum by ~0.023.  Set *presegmented* only
    when the caller genuinely supplies one-sentence-per-line text.
    """
    text = (text or "").strip()
    if not text:
        return ""
    if presegmented:
        return "\n".join(p.strip() for p in text.split("\n") if p.strip())
    text = _WS.sub(" ", text)  # collapse newlines/tabs -> single spaces
    return "\n".join(p.strip() for p in _SENT_SPLIT.split(text) if p.strip())


def _new_scorer(metrics: Sequence[str], use_stemmer: bool):
    try:
        from rouge_score import rouge_scorer
    except Exception as e:  # pragma: no cover
        raise RuntimeError("請先安裝 rouge-score 以計算 ROUGE") from e
    return rouge_scorer.RougeScorer(list(metrics), use_stemmer=use_stemmer)


def score_single(
    pred: str,
    refs: Union[str, Sequence[str]],
    metrics: Sequence[str] = DEFAULT_METRICS,
    use_stemmer: bool = True,
    scorer=None,
    reference_metric: str = "rouge1",
) -> Dict[str, float]:
    """Score one prediction against one *or several* references.

    Multi-reference datasets (e.g. SciTLDR-AIC, whose ``target`` field holds
    alternative TLDRs) must not concatenate references.  This local primitive
    selects one reference by ROUGE-1 and reports every requested metric from
    it; it is intentionally not exposed as the official SciTLDR protocol until
    conformance against the pinned official evaluator has been demonstrated.
    """
    if isinstance(refs, str):
        refs = [refs]
    refs = list(refs)
    if not refs:
        raise ValueError("At least one reference is required")
    scorer_metrics = tuple(dict.fromkeys((*metrics, reference_metric)))
    sc = scorer or _new_scorer(scorer_metrics, use_stemmer)
    p = _as_lsum(pred)
    candidates = [sc.score(_as_lsum(ref), p) for ref in refs]
    chosen = max(candidates, key=lambda scores: scores[reference_metric].fmeasure)
    return {m: chosen[m].fmeasure for m in metrics}


def rouge_scores(
    preds: List[str],
    refs: Sequence[Union[str, Sequence[str]]],
    metrics: Sequence[str] = DEFAULT_METRICS,
    use_stemmer: bool = True,
    return_per_example: bool = False,
    reference_metric: str = "rouge1",
):
    """Corpus-level mean ROUGE.

    Parameters
    ----------
    preds : list of str
    refs  : list of str, or list of list-of-str for multi-reference data
    metrics : defaults to (rouge1, rouge2, **rougeLsum**)

    Returns
    -------
    dict of metric -> mean F-measure, or ``(means, per_example)`` when
    *return_per_example* is True (needed for significance testing).
    """
    if len(preds) != len(refs):
        raise ValueError(
            f"Prediction/reference length mismatch: {len(preds)} != {len(refs)}"
        )
    if not preds:
        raise ValueError("Cannot evaluate an empty corpus")
    scorer_metrics = tuple(dict.fromkeys((*metrics, reference_metric)))
    sc = _new_scorer(scorer_metrics, use_stemmer)
    totals = {m: 0.0 for m in metrics}
    per_example: List[Dict[str, float]] = []
    n = 0
    for p, r in zip(preds, refs):
        s = score_single(
            p,
            r,
            metrics=metrics,
            use_stemmer=use_stemmer,
            scorer=sc,
            reference_metric=reference_metric,
        )
        per_example.append(s)
        for m in metrics:
            totals[m] += s[m]
        n += 1
    means = {m: totals[m] / n for m in metrics}
    if return_per_example:
        return means, per_example
    return means


def rouge_scores_legacy(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """The ORIGINAL (incorrect for multi-sentence) behaviour.

    Kept only so the pre-fix numbers in the ICT Express draft can be
    reproduced for the response letter.  Do not use for new results.
    """
    sc = _new_scorer(("rouge1", "rouge2", "rougeL"), True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = max(1, len(preds))
    for p, r in zip(preds, refs):
        s = sc.score(r, p)
        for k in totals:
            totals[k] += s[k].fmeasure
    return {k: v / n for k, v in totals.items()}
