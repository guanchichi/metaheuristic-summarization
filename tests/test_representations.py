"""Representation-level correctness and failure-boundary tests."""

import numpy as np
import pytest

from src.representations.similarity import cosine_similarity_matrix


def test_cosine_similarity_matches_hand_computed_orthogonal_vectors():
    matrix = cosine_similarity_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
    assert matrix == pytest.approx(np.eye(2))


def test_cosine_similarity_rejects_nonfinite_input_instead_of_falling_back():
    with pytest.raises(ValueError, match="NaN"):
        cosine_similarity_matrix(np.array([[1.0, np.nan], [0.0, 1.0]]))
