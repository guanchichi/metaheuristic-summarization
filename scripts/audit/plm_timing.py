"""Decompose PLM per-document cost into model LOADING vs INFERENCE.

The legacy pipeline called ``from_pretrained`` inside ``_sentence_embeddings``,
i.e. once per document, so its per-article timings largely measured repeated
model construction rather than encoder throughput. That is also the likely
explanation for the ~1.5x BERT/RoBERTa gap Reviewer #4 questioned, since two
architecturally equivalent encoders should not differ much at inference.

This script is a DIAGNOSTIC, not the paper's runtime protocol. Published
timings additionally require: fixed hardware, fixed thread/batch policy,
warm-up exclusion, >=5 repeats, and median/mean/std/P95 reporting.

Usage
-----
    python -m scripts.audit.plm_timing --sentences 40 --repeats 3
"""
from __future__ import annotations

import argparse
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+",
                    default=["bert-base-uncased", "roberta-base", "xlnet-base-cased"])
    ap.add_argument("--sentences", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    import torch
    from src.models.extractive.encoder_rank import (
        load_encoder, clear_encoder_cache, _sentence_embeddings,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    doc = [f"This is sentence number {i} of a moderately long news article "
           f"about an event that involved several people." for i in range(args.sentences)]

    print(f"device = {device}   torch threads = {torch.get_num_threads()}   "
          f"{args.sentences} sentences/doc   batch_size = {args.batch_size}")
    print(f"{'model':22s} {'load ms':>10s} {'infer ms':>10s} {'total ms':>10s} {'load %':>8s}")
    print("-" * 66)

    results = {}
    for name in args.models:
        try:
            clear_encoder_cache()
            t0 = time.perf_counter()
            load_encoder(name, device=device)
            t_load = (time.perf_counter() - t0) * 1000

            _sentence_embeddings(doc[:8], model_name=name, device=device)  # warm-up
            times = []
            for _ in range(args.repeats):
                t1 = time.perf_counter()
                _sentence_embeddings(doc, model_name=name, device=device)
                times.append((time.perf_counter() - t1) * 1000)
            t_inf = sum(times) / len(times)
            total = t_load + t_inf
            results[name] = (t_load, t_inf)
            print(f"{name:22s} {t_load:10.1f} {t_inf:10.1f} {total:10.1f} "
                  f"{100*t_load/total:7.1f}%")
        except Exception as e:
            print(f"{name:22s}  FAILED: {type(e).__name__}: {str(e)[:44]}")

    if "bert-base-uncased" in results and "roberta-base" in results:
        lb, ib = results["bert-base-uncased"]
        lr, ir = results["roberta-base"]
        print()
        print(f"BERT/RoBERTa inference-only ratio : {ib/ir:.2f}x  "
              f"(expect ~1.0 for equivalent encoders)")
        print(f"BERT/RoBERTa load+inference ratio : {(lb+ib)/(lr+ir):.2f}x  "
              f"(the regime the legacy timings were in)")

    clear_encoder_cache()


if __name__ == "__main__":
    main()
