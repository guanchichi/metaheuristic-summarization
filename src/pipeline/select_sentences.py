"""Main sentence-selection pipeline.

Orchestrates feature building, candidate-pool construction, and
optimizer dispatch — each delegated to its own module.
"""

import argparse
import os
import time
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from src.utils.io import (
    load_yaml,
    ensure_dir,
    now_stamp,
    read_jsonl,
    write_jsonl,
    set_global_seed,
)
from src.representations.sent_vectors import SentenceVectors
from src.representations.similarity import cosine_similarity_matrix
from src.data.schemas import flatten_sentence_records, validate_candidate_record

from src.pipeline.feature_builder import build_base_scores
from src.pipeline.candidate_builder import build_candidate_pool
from src.pipeline.optimizer_dispatch import dispatch_optimizer
from src.objectives.factory import build_objective_spec, validate_selector_for_task


# ------------------------------------------------------------------ #
#  Core per-document summarisation                                     #
# ------------------------------------------------------------------ #

def validate_requested_split(doc: Dict, requested_split: str) -> None:
    """Prevent a canonical row from being run under the wrong split label."""

    if "schema_version" in doc and doc.get("split") != requested_split:
        raise ValueError(
            f"canonical row {doc.get('id')!r} belongs to split {doc.get('split')!r}, "
            f"but --split is {requested_split!r}"
        )


def attach_selector_salience(
    candidate_records: List[Dict],
    base_scores: List[float],
    source: str,
) -> List[float]:
    """Resolve the exact auditable score passed from candidates to selector."""

    normalized_source = (source or "base_score").strip().lower()
    selector_scores: List[float] = []
    for candidate in candidate_records:
        index = candidate["original_index"]
        if normalized_source in {"base_score", "membership_only"}:
            value = float(base_scores[index])
        elif normalized_source == "rrf_fusion":
            value = float(candidate["fusion_normalized"])
        elif normalized_source.endswith("_percentile"):
            route = normalized_source[: -len("_percentile")]
            if route not in candidate["route_scores"]:
                raise ValueError(
                    f"selector salience source {source!r} requires route {route!r}"
                )
            value = float(candidate["route_scores"][route]["percentile"])
        else:
            raise ValueError(f"unknown selector.salience_source: {source!r}")
        candidate["selector_salience"] = value
        candidate["selector_salience_source"] = normalized_source
        validate_candidate_record(candidate)
        selector_scores.append(value)
    return selector_scores


def summarize_one(doc: Dict, cfg: Dict) -> Dict:
    sentence_records = flatten_sentence_records(doc)
    sentences: List[str] = [record["text"] for record in sentence_records]

    cand_cfg = cfg.get("candidates", {})
    # A gold-reference-dependent candidate size is an oracle diagnostic, not
    # a deployable inference rule.  Refuse old configs before doing any model
    # or feature work.
    if cand_cfg.get("recall_target") is not None:
        raise ValueError(
            "candidates.recall_target is forbidden in the production selection "
            "pipeline because it requires gold references; run oracle/candidate "
            "recall analysis as a separate diagnostic"
        )

    # 1. Similarity matrix (needed by graph features, candidates, NSGA-II)
    rep_cfg = cfg.get("representations", {})
    sim = None
    if bool(rep_cfg.get("use", True)) and len(sentences) > 0:
        method = rep_cfg.get("method", "tfidf")
        vec = SentenceVectors(method=method)
        X = vec.fit_transform(sentences)
        sim = cosine_similarity_matrix(X)

    # 2. Feature scores
    base_scores = build_base_scores(sentences, cfg, similarity_matrix=sim)

    # 3. Length / redundancy parameters
    lc = cfg.get("length_control", {})
    unit = (lc.get("unit", "tokens") or "tokens").lower()
    if unit not in {"words", "tokens", "sentences"}:
        raise ValueError("length_control.unit must be words, tokens, or sentences")
    max_tokens = int(lc.get("max_tokens", 100))
    max_words = int(lc.get("max_words", 400))
    selector_budget = max_words if unit == "words" else max_tokens
    max_sents_limit = lc.get("max_sentences", None)
    max_sents = int(max_sents_limit) if (max_sents_limit is not None) else None
    alpha = float(cfg.get("redundancy", {}).get("lambda", 0.7))
    objective_spec = build_objective_spec(doc.get("task_profile"), cfg)
    method_opt = cfg.get("optimizer", {}).get("method", "greedy").lower()
    validate_selector_for_task(objective_spec, method_opt)
    required_max_sentences = objective_spec.get("required_max_sentences")
    if required_max_sentences is not None:
        if max_sents is not None and max_sents != required_max_sentences:
            raise ValueError(
                f"task profile requires max_sentences={required_max_sentences}, "
                f"but config declares {max_sents}"
            )
        max_sents = int(required_max_sentences)

    # 4. Candidate pool. Per-route quota and final selector budget are
    # deliberately separate from the output-length budget above.
    budget_cfg = cfg.get("candidate_budget", {})
    if isinstance(budget_cfg, dict):
        route_top_k_budget = budget_cfg.get(
            "route_top_k", budget_cfg.get("per_route", cand_cfg.get("k"))
        )
        min_per_route = budget_cfg.get("min_per_route", 0)
        total_candidate_budget = budget_cfg.get(
            "total", cand_cfg.get("total_budget")
        )
    elif budget_cfg in (None, {}):
        route_top_k_budget = cand_cfg.get("k")
        min_per_route = 0
        total_candidate_budget = cand_cfg.get("total_budget")
    else:
        route_top_k_budget = cand_cfg.get("k")
        min_per_route = 0
        total_candidate_budget = budget_cfg
    k = int(
        min(15, len(sentences))
        if route_top_k_budget is None
        else route_top_k_budget
    )
    total_candidate_budget = (
        int(total_candidate_budget) if total_candidate_budget is not None else None
    )
    use_cand = bool(cand_cfg.get("use", True))
    mode = (cand_cfg.get("mode", "hard") or "hard").lower()
    compute_cfg = cfg.get("compute_budget", {}) or {}
    compute_mode = str(compute_cfg.get("mode", "fixed")).lower()
    if compute_mode != "fixed":
        raise ValueError(
            "only compute_budget.mode='fixed' is implemented; adaptive routing "
            "must not be claimed before its validation-frozen policy exists"
        )
    sources = (
        compute_cfg.get("enabled_routes")
        or cand_cfg.get("sources", ["score"])
        or ["score"]
    )
    soft_boost = float(cand_cfg.get("soft_boost", 1.05))

    g_thresh = float(cfg.get("graph_params", {}).get("threshold", 0.0))
    candidate_pool_result = (
        build_candidate_pool(
            sentence_records,
            base_scores,
            k,
            sources,
            sim_matrix=sim,
            threshold=g_thresh,
            total_budget=total_candidate_budget,
            min_per_route=min_per_route,
            route_config=cfg.get("routes", {}) or {},
            coverage_guard=cfg.get("coverage_guard", {}) or {},
            rrf_constant=int(cand_cfg.get("rrf_constant", 60)),
        )
        if use_cand
        else {
            "records": [],
            "route_proposals": {},
            "allocation": {"actual_size": 0},
        }
    )
    candidate_records = candidate_pool_result["records"]
    cand_idx = [record["original_index"] for record in candidate_records]
    selector_cfg = cfg.get("selector", {}) or {}
    salience_source = str(selector_cfg.get("salience_source", "base_score"))

    # 5. Apply candidate mode
    if use_cand and cand_idx:
        if mode == "hard":
            sub_sentences = [sentences[i] for i in cand_idx]
            sub_scores = attach_selector_salience(
                candidate_records, base_scores, salience_source
            )
            sub_sim = sim[np.ix_(cand_idx, cand_idx)] if sim is not None else None
        else:
            if salience_source.lower() not in {"base_score", "membership_only"}:
                raise ValueError(
                    "provenance-aware selector salience requires candidates.mode='hard'"
                )
            sub_sentences = sentences
            sub_scores = base_scores[:]
            for i in cand_idx:
                sub_scores[i] = float(sub_scores[i]) * soft_boost
            sub_sim = sim
    else:
        sub_sentences = sentences
        sub_scores = base_scores
        sub_sim = sim

    # 6. Optimizer dispatch. A word budget no longer bypasses the configured
    # selector; it is simply the selector's independent output constraint.
    picked_sub = dispatch_optimizer(
        method_opt,
        sub_sentences,
        sub_scores,
        sub_sim,
        selector_budget,
        cfg,
        alpha,
        unit,
        max_sents,
        objective_spec,
    )

    # 7. Map back to original indices
    if use_cand and cand_idx and mode == "hard":
        selected = sorted(cand_idx[i] for i in picked_sub)
    else:
        selected = sorted(picked_sub)

    selected.sort()
    summary_sentences = [sentences[i] for i in selected]
    summary = "\n".join(summary_sentences)
    candidate_by_index = {
        candidate["original_index"]: candidate for candidate in candidate_records
    }
    selected_sentences = [
        {
            **sentence_records[index],
            "selection_order": order,
            "selection_evidence": (
                {
                    "selector_salience": candidate_by_index[index].get(
                        "selector_salience"
                    ),
                    "selector_salience_source": candidate_by_index[index].get(
                        "selector_salience_source"
                    ),
                    "fusion_score": candidate_by_index[index]["fusion_score"],
                    "fusion_normalized": candidate_by_index[index][
                        "fusion_normalized"
                    ],
                    "fused_rank": candidate_by_index[index]["fused_rank"],
                    "route_agreement": candidate_by_index[index]["route_agreement"],
                }
                if index in candidate_by_index
                else None
            ),
        }
        for order, index in enumerate(selected)
    ]
    return {
        "id": doc.get("id"),
        "selected_indices": selected,
        "selected_sentences": selected_sentences,
        "summary_sentences": summary_sentences,
        "summary": summary,
        "candidate_records": candidate_records,
        "candidate_pool": {
            "enabled": use_cand,
            "configured_sources": list(sources) if use_cand else [],
            "route_top_k": k if use_cand else None,
            "min_per_route": min_per_route if use_cand else None,
            "total_cap": total_candidate_budget if use_cand else None,
            "actual_size": len(candidate_records),
            "coverage_guard": dict(cfg.get("coverage_guard", {}) or {}),
            "selector_salience_source": salience_source,
            "route_proposals": candidate_pool_result["route_proposals"],
            "allocation": candidate_pool_result["allocation"],
        },
        "objective_spec": objective_spec,
        "output_budget": {
            "unit": unit,
            "max_words": max_words if unit == "words" else None,
            "max_tokens": max_tokens if unit == "tokens" else None,
            "max_sentences": max_sents,
        },
        "task_profile": doc.get("task_profile"),
    }


# ------------------------------------------------------------------ #
#  CLI entry-point                                                     #
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to config yaml")
    ap.add_argument("--split", required=True, help="dataset split name")
    ap.add_argument("--input", required=True, help="processed jsonl path")
    ap.add_argument("--run_dir", default="runs", help="runs output root")
    ap.add_argument("--stamp", default=None, help="optional fixed stamp for output dir")
    ap.add_argument("--optimizer", default=None, help="override optimizer.method in config")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.optimizer:
        cfg.setdefault("optimizer", {})
        cfg["optimizer"]["method"] = args.optimizer

    # Guard: Stage2 union input should use fast (non-BERT) optimizers only
    method_opt = (cfg.get("optimizer", {}).get("method") or "").lower()
    in_path = str(args.input)
    is_stage2_union = ("stage2" in in_path and "union" in in_path)
    if is_stage2_union and method_opt in ("bert", "roberta", "xlnet", "fused"):
        raise RuntimeError(
            f"Stage2 union input detected ({in_path}). Please use non-BERT optimizers: fast | fast_grasp | fast_nsga2. "
            f"Current optimizer '{method_opt}' is not allowed for Stage2."
        )

    set_global_seed(cfg.get("seed"))
    stamp = args.stamp or now_stamp()
    out_dir = os.path.join(args.run_dir, stamp)
    ensure_dir(out_dir)

    preds_path = os.path.join(out_dir, "predictions.jsonl")
    t0 = time.perf_counter()
    docs = list(read_jsonl(args.input))
    rows = []
    for doc in tqdm(docs, desc="Summarizing", total=len(docs)):
        validate_requested_split(doc, args.split)
        rows.append(summarize_one(doc, cfg))
    write_jsonl(preds_path, rows)
    t1 = time.perf_counter()

    # dump the config used
    import json
    with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    # A formal run is incomplete if its timing artifact cannot be written.
    with open(os.path.join(out_dir, "time_select_seconds.txt"), "w", encoding="utf-8") as f:
        f.write(f"{t1 - t0:.6f}")
    print(f"Wrote predictions to {preds_path}")


if __name__ == "__main__":
    main()
