"""Candidate routing, provenance fusion, and coverage guards.

Every enabled route scores the complete input before its quota is applied.  A
route failure is a failed run: this module never replaces missing scores with
zeros and never silently switches to another route.
"""

from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from src.data.schemas import validate_candidate_record
from src.features.graph import (
    compute_textrank_scores,
    sparse_tfidf_knn_textrank_scores,
)
from src.models.extractive.encoder_rank import encoder_route_scores
from src.representations.tfidf_helper import tfidf_scores_and_sim
from src.utils.tokenizer import count_tokens


RouteMetadata = Dict[str, Any]


def _document_aware_position_scores(sentence_records: List[Dict]) -> List[float]:
    group_sizes: Dict[object, int] = {}
    for record in sentence_records:
        group = record.get("document_id") or "__legacy_flat__"
        group_sizes[group] = group_sizes.get(group, 0) + 1

    scores = []
    for record in sentence_records:
        group = record.get("document_id") or "__legacy_flat__"
        position = int(record.get("document_position", record["original_index"]))
        size = group_sizes[group]
        scores.append(1.0 if size <= 1 else 1.0 - position / (size - 1))
    return scores


def _canonical_route_name(raw_source: str) -> str:
    name = (raw_source or "").strip().lower()
    aliases = {
        "score": "lexical",
        "lexical": "lexical",
        "centrality": "tfidf_centroid",
        "tfidf": "tfidf_centroid",
        "tfidf_centroid": "tfidf_centroid",
        "graph": "graph",
        "textrank": "graph",
        "plm": "semantic",
        "semantic": "semantic",
        # Compatibility only. Position is labelled as a guard signal in the
        # artifact and should be configured through coverage_guard in new runs.
        "position": "position_guard",
        "position_guard": "position_guard",
    }
    if name not in aliases:
        raise ValueError(f"unknown candidate route: {raw_source!r}")
    return aliases[name]


def _config_for_route(
    route_config: Mapping[str, Mapping[str, Any]],
    canonical_name: str,
    raw_source: str,
) -> Mapping[str, Any]:
    value = route_config.get(canonical_name, route_config.get(raw_source, {}))
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration for route {canonical_name!r} must be an object")
    return value


def _score_route(
    route: str,
    sentences: List[str],
    sentence_records: List[Dict],
    base_scores: List[float],
    sim_matrix,
    threshold: float,
    config: Mapping[str, Any],
) -> Tuple[List[float], RouteMetadata]:
    n = len(sentences)
    if route == "lexical":
        values = [float(value) for value in base_scores]
        metadata: RouteMetadata = {
            "route_type": "lexical_handcrafted",
            "model_revision": str(config.get("revision", "base_scores:v1")),
            "estimated_cost": {"scored_sentences": n},
        }
    elif route == "position_guard":
        values = _document_aware_position_scores(sentence_records)
        metadata = {
            "route_type": "coverage_guard_signal",
            "model_revision": "document_position:v1",
            "estimated_cost": {"scored_sentences": n},
        }
    elif route == "tfidf_centroid":
        values, _ = tfidf_scores_and_sim(
            sentences,
            sublinear_tf=bool(config.get("sublinear_tf", False)),
            ngram_range=tuple(config.get("ngram_range", (1, 1))),
        )
        metadata = {
            "route_type": "lexical_tfidf_diagnostic",
            "model_revision": "sklearn_tfidf_centroid:v1",
            "estimated_cost": {"vectorized_sentences": n},
        }
    elif route == "graph":
        implementation = str(config.get("implementation", "sparse_knn")).lower()
        if implementation == "sparse_knn":
            values, graph_facts = sparse_tfidf_knn_textrank_scores(
                sentences,
                n_neighbors=int(config.get("n_neighbors", 8)),
                min_similarity=float(config.get("min_similarity", threshold or 0.05)),
                alpha=float(config.get("alpha", 0.85)),
                max_iter=int(config.get("max_iter", 100)),
                tol=float(config.get("tol", 1e-6)),
            )
            metadata = {
                "route_type": "graph_centrality",
                "model_revision": "sparse_tfidf_knn_textrank:v1",
                "estimated_cost": graph_facts,
                "representation": "tfidf_cosine_knn",
                "bounded_sparse": True,
            }
        elif implementation == "dense_legacy":
            if sim_matrix is None:
                _, graph_sim = tfidf_scores_and_sim(sentences)
                representation = "tfidf_internal"
            else:
                graph_sim = sim_matrix
                representation = str(
                    config.get("representation", "pipeline_similarity")
                )
            values = [
                float(value)
                for value in compute_textrank_scores(graph_sim, threshold=threshold)
            ]
            metadata = {
                "route_type": "graph_centrality",
                "model_revision": "dense_textrank_legacy:v1",
                "estimated_cost": {
                    "nodes": n,
                    "dense_similarity_entries": n * n,
                },
                "representation": representation,
                "threshold": threshold,
                "bounded_sparse": False,
            }
        else:
            raise ValueError(
                "routes.graph.implementation must be 'sparse_knn' or 'dense_legacy'"
            )
    elif route == "semantic":
        model_name = config.get("model_name")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                "semantic candidate route requires routes.semantic.model_name; "
                "use an explicit sentence-similarity checkpoint"
            )
        values, semantic_metadata = encoder_route_scores(
            sentences,
            model_name=model_name,
            device=config.get("device"),
            batch_size=int(config.get("batch_size", 16)),
            max_model_tokens=int(config.get("max_model_tokens", 256)),
            revision=config.get("revision"),
        )
        metadata = {
            "route_type": "semantic_sentence_encoder",
            **semantic_metadata,
        }
    else:  # pragma: no cover - guarded by _canonical_route_name
        raise ValueError(f"unsupported candidate route: {route!r}")

    if len(values) != n:
        raise RuntimeError(
            f"candidate route {route!r} returned {len(values)} scores for {n} rows"
        )
    return [float(value) for value in values], metadata


def _rank(values: List[float]) -> Tuple[List[int], Dict[int, int]]:
    ranking = sorted(range(len(values)), key=lambda index: (-values[index], index))
    return ranking, {index: rank for rank, index in enumerate(ranking, start=1)}


def _coverage_guard_indices(
    sentence_records: List[Dict],
    fused_ranks: Mapping[int, int],
    guard_config: Mapping[str, Any],
) -> Dict[int, List[str]]:
    """Return deterministic guard nominees and their auditable reasons."""

    if not guard_config or not bool(guard_config.get("enabled", False)):
        return {}

    reasons: Dict[int, List[str]] = {}

    def add_best_by(field: str, reason: str) -> None:
        groups: Dict[object, List[int]] = {}
        for index, record in enumerate(sentence_records):
            key = record.get(field)
            if key is not None:
                groups.setdefault(key, []).append(index)
        for indices in groups.values():
            best = min(indices, key=lambda index: (fused_ranks[index], index))
            reasons.setdefault(best, []).append(reason)

    if bool(guard_config.get("document", True)):
        add_best_by("document_id", "guard:document")
    if bool(guard_config.get("section", False)):
        add_best_by("section_id", "guard:section")
    if bool(guard_config.get("position", False)):
        groups: Dict[object, List[int]] = {}
        for index, record in enumerate(sentence_records):
            key = record.get("document_id") or "__legacy_flat__"
            groups.setdefault(key, []).append(index)
        for indices in groups.values():
            lead = min(
                indices,
                key=lambda index: (
                    int(sentence_records[index].get("document_position", index)),
                    index,
                ),
            )
            reasons.setdefault(lead, []).append("guard:position")

    max_items = guard_config.get("max_items")
    if max_items is not None:
        limit = max(0, int(max_items))
        ordered = sorted(reasons, key=lambda index: (fused_ranks[index], index))[:limit]
        reasons = {index: reasons[index] for index in ordered}
    return reasons


def build_candidate_records(
    sentence_records: List[Dict],
    base_scores: List[float],
    k: int,
    sources: List[str],
    sim_matrix=None,
    threshold: float = 0.0,
    *,
    total_budget: Optional[int] = None,
    route_config: Optional[Mapping[str, Mapping[str, Any]]] = None,
    coverage_guard: Optional[Mapping[str, Any]] = None,
    rrf_constant: int = 60,
) -> List[Dict]:
    """Score full input, fuse route provenance, and enforce one total budget.

    ``k`` is the per-route quota. ``total_budget`` is the final number of
    candidates passed to the selector. When a total budget is supplied the
    pool is deterministically filled (or capped) to ``min(total_budget, N)``.
    """

    n = len(sentence_records)
    if n == 0:
        return []
    if len(base_scores) != n:
        raise ValueError("base_scores length must match sentence_records")
    if isinstance(sources, (str, bytes)) or not sources:
        raise ValueError("at least one candidate route is required")
    if rrf_constant < 1:
        raise ValueError("rrf_constant must be positive")

    quota = int(k)
    if quota < 1:
        raise ValueError("per-route candidate quota must be positive")
    quota = min(quota, n)
    if total_budget is not None:
        total_budget = int(total_budget)
        if total_budget < 1:
            raise ValueError("total candidate budget must be positive")
        total_budget = min(total_budget, n)
    sentences = [record["text"] for record in sentence_records]
    configs = route_config or {}

    route_values: Dict[str, List[float]] = {}
    route_metadata: Dict[str, RouteMetadata] = {}
    for raw_source in sources:
        route = _canonical_route_name(raw_source)
        if route in route_values:
            continue
        config = _config_for_route(configs, route, raw_source)
        try:
            values, metadata = _score_route(
                route,
                sentences,
                sentence_records,
                base_scores,
                sim_matrix,
                threshold,
                config,
            )
        except Exception as exc:
            raise RuntimeError(f"candidate route {route!r} failed: {exc}") from exc
        route_values[route] = values
        route_metadata[route] = metadata

    route_rankings: Dict[str, List[int]] = {}
    route_ranks: Dict[str, Dict[int, int]] = {}
    union_indices: Set[int] = set()
    for route, values in route_values.items():
        ranking, ranks = _rank(values)
        route_rankings[route] = ranking
        route_ranks[route] = ranks
        union_indices.update(ranking[:quota])

    fusion_scores = {
        index: sum(
            1.0 / (rrf_constant + route_ranks[route][index])
            for route in route_values
        )
        for index in range(n)
    }
    fused_order = sorted(range(n), key=lambda i: (-fusion_scores[i], i))
    fused_ranks = {index: rank for rank, index in enumerate(fused_order, start=1)}
    guard_reasons = _coverage_guard_indices(
        sentence_records, fused_ranks, coverage_guard or {}
    )

    inclusion_reasons: Dict[int, List[str]] = {
        index: [
            f"route:{route}"
            for route, ranking in route_rankings.items()
            if index in ranking[:quota]
        ]
        for index in union_indices
    }
    for index, reasons in guard_reasons.items():
        inclusion_reasons.setdefault(index, []).extend(reasons)

    if total_budget is None:
        selected_indices = set(inclusion_reasons)
    else:
        # Guards are reserved first, but are themselves ranked by the same
        # inference-only fused evidence when there are more guards than slots.
        guarded = sorted(guard_reasons, key=lambda i: (fused_ranks[i], i))
        chosen = guarded[:total_budget]
        for index in fused_order:
            if len(chosen) >= total_budget:
                break
            if index in chosen:
                continue
            chosen.append(index)
            inclusion_reasons.setdefault(index, []).append("budget_fill")
        selected_indices = set(chosen)

    candidates = []
    for index in sorted(selected_indices):
        sentence = sentence_records[index]
        candidate = {
            "sentence_id": sentence["sentence_id"],
            "text": sentence["text"],
            "original_index": sentence["original_index"],
            "document_id": sentence.get("document_id"),
            "section_id": sentence.get("section_id"),
            "word_count": len(sentence["text"].split()),
            "route_scores": {},
            "selected_by_routes": [],
            "fusion_score": fusion_scores[index],
            "fused_rank": fused_ranks[index],
            "inclusion_reasons": inclusion_reasons.get(index, []),
        }
        for route, values in route_values.items():
            rank = route_ranks[route][index]
            metadata = route_metadata[route]
            candidate["route_scores"][route] = {
                "raw": values[index],
                "rank": rank,
                "percentile": 1.0 if n == 1 else 1.0 - ((rank - 1) / (n - 1)),
                "route_type": metadata["route_type"],
                "model_revision": metadata.get("model_revision"),
                "estimated_cost": metadata.get("estimated_cost"),
                "metadata": {
                    key: value
                    for key, value in metadata.items()
                    if key
                    not in {"route_type", "model_revision", "estimated_cost"}
                },
            }
            if rank <= quota:
                candidate["selected_by_routes"].append(route)
        candidate["route_agreement"] = len(candidate["selected_by_routes"])
        validate_candidate_record(candidate)
        candidates.append(candidate)
    return candidates


def build_candidate_union(
    sentences: List[str],
    base_scores: List[float],
    k: int,
    sources: List[str],
    sim_matrix=None,
    threshold: float = 0.0,
) -> List[int]:
    """Backward-compatible strict wrapper returning candidate indices only."""

    records = [
        {
            "sentence_id": f"legacy:s{index:06d}",
            "text": sentence,
            "original_index": index,
            "document_id": None,
            "section_id": None,
            "document_position": index,
        }
        for index, sentence in enumerate(sentences)
    ]
    return [
        record["original_index"]
        for record in build_candidate_records(
            records,
            base_scores,
            k,
            sources,
            sim_matrix=sim_matrix,
            threshold=threshold,
        )
    ]


def greedy_oracle_indices(
    sentences: List[str], reference: str, max_tokens: int
) -> List[int]:
    """Greedy reference oracle for offline diagnostic analysis only."""
    try:
        from rouge_score import rouge_scorer
    except Exception as exc:
        raise RuntimeError(
            "greedy oracle diagnostics require the 'rouge-score' package"
        ) from exc
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    selected: List[int] = []
    cur_summary = ""
    cur_tokens = 0
    best_f = 0.0
    n = len(sentences)
    for _ in range(n):
        best_i = None
        best_gain = 0.0
        for i in range(n):
            if i in selected:
                continue
            t = count_tokens(sentences[i])
            if cur_tokens + t > max_tokens:
                continue
            cand = (cur_summary + " " + sentences[i]).strip()
            f = scorer.score(reference or "", cand)["rouge1"].fmeasure
            gain = f - best_f
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        cur_summary = (cur_summary + " " + sentences[best_i]).strip()
        cur_tokens += count_tokens(sentences[best_i])
        best_f = scorer.score(reference or "", cur_summary)["rouge1"].fmeasure
    return sorted(selected)
