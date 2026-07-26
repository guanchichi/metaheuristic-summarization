"""Task-profile objective and aggregation contracts."""

import numpy as np
import pytest

from src.objectives.factory import aggregate_importance, build_objective_spec


def test_mean_importance_removes_subset_size_reward():
    importance = np.array([1.0, 1.0, 0.0])
    sentences = ["one", "two", "three"]
    assert aggregate_importance(
        importance, np.array([0]), sentences, "mean"
    ) == pytest.approx(1.0)
    assert aggregate_importance(
        importance, np.array([0, 1]), sentences, "mean"
    ) == pytest.approx(1.0)


def test_unknown_coverage_method_fails_instead_of_using_max():
    with pytest.raises(ValueError, match="unknown coverage method"):
        build_objective_spec(
            {"input_mode": "single_document", "output_mode": "multi_sentence"},
            {"objectives": {"coverage_method": "typo"}},
        )


def test_group_coverage_cannot_be_claimed_before_implementation():
    with pytest.raises(NotImplementedError, match="not implemented"):
        build_objective_spec(
            {"input_mode": "multi_document", "output_mode": "multi_sentence"},
            {"objectives": {"group_coverage": True}},
        )
