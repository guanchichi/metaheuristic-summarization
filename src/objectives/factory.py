"""Create only objectives that are meaningful for the declared task profile."""

from typing import Any, Dict, Mapping, Optional

import numpy as np

from src.utils.tokenizer import count_tokens


SINGLE_SENTENCE_SELECTORS = frozenset(
    {"greedy", "bert", "roberta", "xlnet", "fast", "fast_fused", "tfidf_fused"}
)


def aggregate_importance(
    importance: np.ndarray,
    indices: np.ndarray,
    sentences: list[str],
    method: str,
) -> float:
    """Aggregate salience without silently introducing subset-size reward."""

    if indices.size == 0:
        return 0.0
    if method == "sum":
        return float(np.sum(importance[indices]))
    if method == "mean":
        return float(np.mean(importance[indices]))
    if method == "length_normalized":
        words = sum(max(1, count_tokens(sentences[index])) for index in indices)
        return float(np.sum(importance[indices]) / words)
    raise ValueError(f"unknown importance aggregation: {method!r}")


def build_objective_spec(
    task_profile: Optional[Mapping[str, Any]], cfg: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return an auditable objective matrix without reading a reference."""

    objective_cfg = cfg.get("objectives", {}) or {}
    if not isinstance(objective_cfg, Mapping):
        raise ValueError("objectives configuration must be an object")

    if not task_profile:
        # Legacy flat rows have no reliable task semantics. Preserve historical
        # behavior, but label it so it cannot be reported as target architecture.
        return {
            "status": "legacy_unprofiled",
            "input_mode": None,
            "output_mode": None,
            "active": ["salience", "facility_coverage", "redundancy"],
            "importance_aggregation": str(
                objective_cfg.get("importance_aggregation", "sum")
            ),
            "coverage_method": str(objective_cfg.get("coverage_method", "max")),
            "coverage_scope": "legacy_candidate_or_source_unprofiled",
            "weights": {
                "salience": float(objective_cfg.get("lambda_importance", 1.0)),
                "facility_coverage": float(objective_cfg.get("lambda_coverage", 0.8)),
                "redundancy": float(objective_cfg.get("lambda_redundancy", 0.7)),
            },
            "required_max_sentences": None,
            "group_coverage": False,
        }

    input_mode = task_profile.get("input_mode")
    output_mode = task_profile.get("output_mode")
    if input_mode not in {"single_document", "multi_document"}:
        raise ValueError(f"invalid task_profile.input_mode: {input_mode!r}")
    if output_mode not in {"single_sentence", "multi_sentence"}:
        raise ValueError(f"invalid task_profile.output_mode: {output_mode!r}")

    group_coverage = bool(objective_cfg.get("group_coverage", False))
    if group_coverage:
        if input_mode != "multi_document":
            raise ValueError("group coverage is only valid for multi-document tasks")
        raise NotImplementedError(
            "document-group coverage is declared but not implemented in the selector; "
            "refusing to report a nonexistent objective"
        )

    if output_mode == "single_sentence":
        return {
            "status": "task_profiled_v1",
            "input_mode": input_mode,
            "output_mode": output_mode,
            "applicable": ["salience"],
            "active": ["salience"],
            "importance_aggregation": "single_item",
            "coverage_method": "max",
            "coverage_scope": "not_applicable",
            "weights": {
                "salience": 1.0,
                "facility_coverage": 0.0,
                "redundancy": 0.0,
            },
            "required_max_sentences": 1,
            "group_coverage": False,
        }

    aggregation = str(objective_cfg.get("importance_aggregation", "mean")).lower()
    if aggregation not in {"mean", "length_normalized"}:
        raise ValueError(
            "profiled multi-sentence importance_aggregation must be 'mean' "
            "or 'length_normalized'; raw 'sum' has candidate-cardinality bias"
        )
    coverage_method = str(objective_cfg.get("coverage_method", "max")).lower()
    if coverage_method not in {"max", "set", "diversity"}:
        raise ValueError(f"unknown coverage method: {coverage_method!r}")
    if coverage_method == "diversity":
        raise ValueError(
            "profiled coverage_method='diversity' mixes redundancy into the "
            "coverage objective and double-counts it; use facility coverage "
            "('max' or its equivalent 'set')"
        )
    applicable = ["salience", "facility_coverage", "redundancy"]
    active = applicable
    return {
        "status": "task_profiled_v1",
        "input_mode": input_mode,
        "output_mode": output_mode,
        "applicable": applicable,
        "active": active,
        "importance_aggregation": aggregation,
        "coverage_method": coverage_method,
        "coverage_scope": "full_source_sentences",
        "weights": {
            "salience": float(objective_cfg.get("lambda_importance", 1.0)),
            "facility_coverage": float(objective_cfg.get("lambda_coverage", 0.8)),
            "redundancy": float(objective_cfg.get("lambda_redundancy", 0.7)),
        },
        "required_max_sentences": None,
        "group_coverage": False,
    }


def validate_selector_for_task(objective_spec: Mapping[str, Any], method: str) -> None:
    """Reject subset/metaheuristic search when the task outputs one sentence."""

    normalized = (method or "").lower()
    if objective_spec.get("output_mode") != "single_sentence":
        return
    if normalized not in SINGLE_SENTENCE_SELECTORS:
        raise ValueError(
            f"single-sentence tasks require deterministic ranking; selector "
            f"{method!r} is not allowed"
        )
