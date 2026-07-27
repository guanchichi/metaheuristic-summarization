"""Deterministic greedy search over the shared selection objective."""

from typing import List, Optional

import numpy as np

from src.objectives.evaluator import (
    ObjectiveWeights,
    SelectionConstraints,
    SelectionObjective,
)


def greedy_select(
    sentences: List[str],
    base_scores: List[float],
    sim_mat: Optional[np.ndarray],
    max_tokens: int,
    alpha: float = 0.7,
    unit: str = "tokens",
    max_sentences: int | None = None,
    *,
    evaluator: SelectionObjective | None = None,
    importance_aggregation: str = "sum",
    coverage_method: str = "max",
    lambda_importance: float | None = None,
    lambda_coverage: float = 0.0,
    lambda_redundancy: float | None = None,
    min_words: int = 0,
    require_nonempty: bool = True,
) -> List[int]:
    """Select a subset by deterministic marginal utility.

    ``alpha`` remains only as a backward-compatible way to derive salience and
    redundancy weights when explicit lambdas are not supplied.  Formal
    pipeline runs inject the same evaluator used by GRASP and NSGA-II.
    """

    if not sentences:
        return []
    if evaluator is None:
        evaluator = SelectionObjective(
            sentences,
            base_scores,
            sim_mat,
            importance_aggregation=importance_aggregation,
            coverage_method=coverage_method,
            weights=ObjectiveWeights(
                salience=(alpha if lambda_importance is None else lambda_importance),
                facility_coverage=lambda_coverage,
                redundancy=(
                    (1.0 - alpha)
                    if lambda_redundancy is None
                    else lambda_redundancy
                ),
            ),
            constraints=SelectionConstraints(
                length_unit=unit,
                max_length=max_tokens,
                min_words=min_words,
                max_sentences=max_sentences,
                require_nonempty=require_nonempty,
            ),
        )

    selected: List[int] = []
    remaining = set(range(len(sentences)))
    while remaining:
        current = evaluator.evaluate(selected)
        ranked: list[tuple[float, int]] = []
        for candidate in sorted(remaining):
            if not evaluator.can_add(selected, candidate):
                continue
            value = evaluator.evaluate(selected + [candidate]).scalar_utility
            ranked.append((value, candidate))
        if not ranked:
            break
        # Stable scientific tie-break: lower candidate index wins.
        best_value, best_candidate = max(ranked, key=lambda item: (item[0], -item[1]))
        # Once all lower bounds are satisfied, do not add a sentence that
        # worsens the declared objective merely to fill the budget.
        if selected and current.feasible and best_value <= current.scalar_utility + 1e-12:
            break
        selected.append(best_candidate)
        remaining.remove(best_candidate)

    evaluator.assert_feasible(selected)
    return sorted(selected)
