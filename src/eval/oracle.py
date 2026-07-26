"""Extractive oracle-reference computation.

WHY THIS FILE EXISTS
--------------------
The ICT Express draft reported an "oracle ROUGE-1 of 0.136" on SciTLDR-AIC
while the system scored 0.234, and Reviewer #4 correctly flagged this as a
logical impossibility (an extractive oracle upper-bounds every extractive
method).

The audit traced that number: it is the mean of SciTLDR's own
``rouge_scores`` field taken over *every source sentence of every document*.
That field holds the ROUGE of each **individual sentence** against the target
-- it exists to derive extractive training labels (``source_labels``).  Its
mean is the expected score of picking ONE SENTENCE AT RANDOM, which is a
lower-bound-flavoured statistic, not an upper bound.

    reproduced mean of the rouge_scores field = 0.13758   (draft said 0.136)
    mean of the best SINGLE sentence per doc  = 0.43106

So there was never a contradiction -- just the wrong column.  This module
computes reproducible oracle references.  Only exhaustive search under the
same constraints is a strict upper bound; greedy search is not.

DEFINITION USED (state this explicitly in the paper)
----------------------------------------------------
Greedy oracle: starting from the empty set, repeatedly add the sentence that
maximises the target metric of the resulting summary, stopping when no
remaining sentence yields a positive gain or the budget is exhausted.  This
is the standard construction (Nallapati et al. 2017; Liu & Lapata 2019).  It
is a *lower bound on the true (exhaustive) oracle*.  It must be labelled
"greedy oracle", never "exact upper bound".

The generic evaluator uses Google ``rouge_score``.  SciTLDR paper-table
conformance still requires the official ``files2rouge`` script and its
max-ROUGE-1 reference-selection rule; this module alone is not a conformance
test.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Sequence, Union

from src.eval.rouge import _as_lsum, _new_scorer, DEFAULT_METRICS


def _fmeasure(scorer, pred: str, refs: Sequence[str], metric: str) -> float:
    p = _as_lsum(pred)
    return max(scorer.score(_as_lsum(r), p)[metric].fmeasure for r in refs)


def greedy_oracle_summary(
    sentences: List[str],
    refs: Union[str, Sequence[str]],
    max_tokens: int | None = None,
    max_sentences: int | None = None,
    metric: str = "rouge1",
    scorer=None,
) -> List[int]:
    """Return the indices selected by the greedy oracle.

    ``max_tokens`` is a legacy API name.  The implementation counts
    whitespace-delimited words, not model tokens.  Exactly one of the word
    budget / sentence budget is normally used; if both are given, both apply.
    """
    if isinstance(refs, str):
        refs = [refs]
    sc = scorer or _new_scorer((metric,), True)

    selected: List[int] = []
    cur = ""
    tot = 0
    best = 0.0
    while True:
        if max_sentences is not None and len(selected) >= max_sentences:
            break
        best_i, best_gain = None, 1e-9
        for i, s in enumerate(sentences):
            if i in selected:
                continue
            w = len(s.split())
            if max_tokens is not None and tot + w > max_tokens:
                continue
            cand = (cur + " " + s).strip()
            f = _fmeasure(sc, cand, refs, metric)
            if f - best > best_gain:
                best_gain, best_i = f - best, i
        if best_i is None:
            break
        selected.append(best_i)
        cur = (cur + " " + sentences[best_i]).strip()
        tot += len(sentences[best_i].split())
        best = _fmeasure(sc, cur, refs, metric)
    return sorted(selected)


def oracle_scores(
    docs: List[Dict],
    max_tokens: int | None = None,
    max_sentences: int | None = None,
    target_metric: str = "rouge1",
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> Dict[str, float]:
    """Corpus-level greedy oracle reference.

    ``highlights`` may be a str or a list of alternative references.
    """
    from src.eval.rouge import rouge_scores

    sel_scorer = _new_scorer((target_metric,), True)
    preds, refs = [], []
    for d in docs:
        sents = d.get("sentences", [])
        ref = d.get("highlights", "")
        idx = greedy_oracle_summary(
            sents, ref, max_tokens=max_tokens, max_sentences=max_sentences,
            metric=target_metric, scorer=sel_scorer,
        )
        preds.append(" ".join(sents[i] for i in idx))
        refs.append(ref)
    return rouge_scores(preds, refs, metrics=metrics)


def main():
    ap = argparse.ArgumentParser(description="Compute a greedy extractive oracle reference")
    ap.add_argument("--input", required=True, help="processed jsonl")
    ap.add_argument(
        "--max_words", "--max_tokens", dest="max_words", type=int, default=None,
        help="whitespace-delimited word budget; --max_tokens is a legacy alias",
    )
    ap.add_argument("--max_sentences", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--target_metric", default="rouge1",
                    help="metric the greedy search optimises (rouge1 is standard)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    docs = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
                if args.limit and len(docs) >= args.limit:
                    break

    m = oracle_scores(
        docs,
        max_tokens=args.max_words,
        max_sentences=args.max_sentences,
        target_metric=args.target_metric,
    )
    print(f"Greedy extractive oracle over {len(docs)} docs "
          f"(max_words={args.max_words}, max_sentences={args.max_sentences}):")
    for k, v in m.items():
        print(f"  {k:12s} {v:.4f}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"n_docs": len(docs), "max_words": args.max_words,
                       "max_sentences": args.max_sentences, "scores": m}, f, indent=2)


if __name__ == "__main__":
    main()
