"""F-0: ID-matched comparison of a system run against a local Lead baseline.

Everything is scored through the project's own evaluator (``src.eval.rouge``),
so both sides share preprocessing, sentence segmentation and ROUGE settings.
This is the check the ICT Express draft never did -- its Table 7 Lead/LexRank/
TextRank numbers were "adopted from [16]".

Usage
-----
    python -m scripts.audit.lead_vs_system \
        --data data/processed/multi_news_test.jsonl \
        --pred runs/tuning_experiments/ExpB_K20_Max_Coverage/predictions.jsonl \
        --budget 245

Note: ``--budget`` counts whitespace-delimited WORDS, matching the legacy
pipeline's "max_tokens" semantics. It is not a model-tokenizer budget.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

from src.eval.rouge import rouge_scores


def load_jsonl(path: str, limit: int | None = None) -> List[Dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
                if limit and len(out) >= limit:
                    break
    return out


def norm_sentences(doc: Dict) -> List[str]:
    ss = doc.get("sentences") or []
    return [s["text"] if isinstance(s, dict) else s for s in ss]


def reference_of(doc: Dict):
    return doc.get("highlights") or doc.get("reference", "")


def lead_prefix(sentences: List[str], max_words: int | None = None,
                max_sentences: int | None = None) -> str:
    if max_sentences is not None:
        return " ".join(sentences[:max_sentences])
    out, tot = [], 0
    for s in sentences:
        w = len(s.split())
        if max_words is not None and tot + w > max_words:
            break
        out.append(s)
        tot += w
    return " ".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="processed jsonl with full sentences")
    ap.add_argument("--pred", required=True, help="a run's predictions.jsonl")
    ap.add_argument("--budget", type=int, default=None,
                    help="Lead word budget (whitespace words)")
    ap.add_argument("--lead_sentences", type=int, default=None,
                    help="use Lead-N sentences instead of a word budget")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    docs = {str(d["id"]): d for d in load_jsonl(args.data, args.limit)}
    order = [str(d["id"]) for d in load_jsonl(args.data, args.limit)]
    preds = {str(r["id"]): r for r in load_jsonl(args.pred)}
    ids = [i for i in order if i in preds]
    if not ids:
        raise SystemExit("No overlapping ids between --data and --pred")

    refs = [reference_of(docs[i]) for i in ids]
    sys_sum = [preds[i]["summary"] for i in ids]
    mean_len = sum(len(s.split()) for s in sys_sum) / len(sys_sum)

    rows = [("System run", sys_sum)]
    if args.lead_sentences is not None:
        rows.append((f"Lead-{args.lead_sentences}",
                     [lead_prefix(norm_sentences(docs[i]),
                                  max_sentences=args.lead_sentences) for i in ids]))
    if args.budget is not None:
        rows.append((f"Lead, {args.budget}-word budget",
                     [lead_prefix(norm_sentences(docs[i]), max_words=args.budget) for i in ids]))
    rows.append(("Lead, per-doc length-matched",
                 [lead_prefix(norm_sentences(docs[i]),
                              max_words=max(1, len(preds[i]["summary"].split()))) for i in ids]))

    print(f"matched docs = {len(ids)}   mean system length = {mean_len:.1f} words\n")
    print(f"{'system':36s} {'R-1':>8s} {'R-2':>8s} {'R-Lsum':>8s} {'words':>7s}")
    print("-" * 72)
    results = {}
    for name, ps in rows:
        m = rouge_scores(ps, refs)
        L = sum(len(p.split()) for p in ps) / len(ps)
        results[name] = m
        print(f"{name:36s} {m['rouge1']:8.4f} {m['rouge2']:8.4f} {m['rougeLsum']:8.4f} {L:7.1f}")

    base = results["System run"]
    print("\nSystem minus each Lead variant (negative = system loses):")
    for name, m in results.items():
        if name == "System run":
            continue
        print(f"  vs {name:34s} "
              f"R-1 {base['rouge1']-m['rouge1']:+.4f}  "
              f"R-2 {base['rouge2']-m['rouge2']:+.4f}  "
              f"R-Lsum {base['rougeLsum']-m['rougeLsum']:+.4f}")
    print("\nNOTE: no paired significance test here. Small deltas are not "
          "evidence of a win; run paired bootstrap before claiming anything.")


if __name__ == "__main__":
    main()
