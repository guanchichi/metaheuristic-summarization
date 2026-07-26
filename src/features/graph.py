"""Dense legacy TextRank and bounded sparse kNN graph utilities."""

from typing import Dict, List, Tuple

import numpy as np


def compute_textrank_scores(
    similarity_matrix: np.ndarray,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
    threshold: float = 0.0,
) -> List[float]:
    """Compute PageRank scores from a dense similarity matrix.

    This function remains for legacy objectives. New long-document candidate
    routes should use :func:`sparse_tfidf_knn_textrank_scores` so the graph is
    bounded by ``O(N * k)`` stored edges rather than a dense ``N x N`` array.
    """

    matrix = np.asarray(similarity_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("similarity_matrix must be square")
    if np.any(~np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("similarity_matrix must contain finite non-negative values")
    n = matrix.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    matrix = matrix.copy()
    if threshold > 0:
        matrix[matrix < threshold] = 0.0

    row_sums = matrix.sum(axis=1)
    non_dangling = row_sums > 0
    transition = np.zeros_like(matrix)
    transition[non_dangling] = (
        matrix[non_dangling] / row_sums[non_dangling, np.newaxis]
    )

    scores = np.ones(n, dtype=float) / n
    teleport = np.ones(n, dtype=float) / n
    for _ in range(max_iter):
        dangling_mass = scores[~non_dangling].sum() / n
        updated = alpha * (transition.T @ scores + dangling_mass) + (1 - alpha) * teleport
        if np.linalg.norm(updated - scores, 1) < tol:
            scores = updated
            break
        scores = updated
    return scores.tolist()


def build_sparse_tfidf_knn_graph(
    sentences: List[str],
    *,
    n_neighbors: int = 8,
    min_similarity: float = 0.05,
) -> Tuple[object, Dict[str, int | float]]:
    """Build a symmetric sparse TF-IDF cosine kNN adjacency matrix."""

    if n_neighbors < 1:
        raise ValueError("n_neighbors must be positive")
    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be in [0, 1]")

    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    n = len(sentences)
    empty_facts: Dict[str, int | float] = {
        "nodes": n,
        "directed_edges_before_symmetry": 0,
        "stored_edges": 0,
        "n_neighbors": n_neighbors,
        "min_similarity": min_similarity,
    }
    if n <= 1:
        return sparse.csr_matrix((n, n), dtype=float), empty_facts

    vectors = TfidfVectorizer(lowercase=True).fit_transform(sentences)
    query_neighbors = min(n, n_neighbors + 1)
    nearest = NearestNeighbors(
        n_neighbors=query_neighbors,
        metric="cosine",
        algorithm="brute",
    ).fit(vectors)
    distances, indices = nearest.kneighbors(vectors, return_distance=True)

    rows: List[int] = []
    columns: List[int] = []
    values: List[float] = []
    for source, (neighbor_indices, neighbor_distances) in enumerate(
        zip(indices, distances)
    ):
        kept = 0
        for target, distance in zip(neighbor_indices, neighbor_distances):
            target = int(target)
            if target == source:
                continue
            similarity = max(0.0, 1.0 - float(distance))
            if similarity < min_similarity:
                continue
            rows.append(source)
            columns.append(target)
            values.append(similarity)
            kept += 1
            if kept >= n_neighbors:
                break

    directed_edges = len(values)
    adjacency = sparse.csr_matrix((values, (rows, columns)), shape=(n, n))
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.eliminate_zeros()
    return adjacency, {
        "nodes": n,
        "directed_edges_before_symmetry": directed_edges,
        "stored_edges": int(adjacency.nnz),
        "n_neighbors": n_neighbors,
        "min_similarity": min_similarity,
    }


def compute_sparse_textrank_scores(
    adjacency,
    *,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> List[float]:
    """Compute PageRank without materializing a dense transition matrix."""

    from scipy import sparse

    matrix = sparse.csr_matrix(adjacency, dtype=float)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("sparse adjacency must be square")
    if matrix.data.size and (
        np.any(~np.isfinite(matrix.data)) or np.any(matrix.data < 0)
    ):
        raise ValueError("sparse adjacency must contain finite non-negative values")
    n = matrix.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    non_dangling = row_sums > 0
    inverse = np.zeros(n, dtype=float)
    inverse[non_dangling] = 1.0 / row_sums[non_dangling]
    transition = sparse.diags(inverse) @ matrix

    scores = np.ones(n, dtype=float) / n
    teleport = np.ones(n, dtype=float) / n
    for _ in range(max_iter):
        dangling_mass = scores[~non_dangling].sum() / n
        updated = alpha * (
            np.asarray(transition.T @ scores).ravel() + dangling_mass
        ) + (1 - alpha) * teleport
        if np.linalg.norm(updated - scores, 1) < tol:
            scores = updated
            break
        scores = updated
    return scores.tolist()


def sparse_tfidf_knn_textrank_scores(
    sentences: List[str],
    *,
    n_neighbors: int = 8,
    min_similarity: float = 0.05,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[List[float], Dict[str, int | float]]:
    """Build the bounded graph and return TextRank scores plus graph facts."""

    adjacency, metadata = build_sparse_tfidf_knn_graph(
        sentences,
        n_neighbors=n_neighbors,
        min_similarity=min_similarity,
    )
    scores = compute_sparse_textrank_scores(
        adjacency, alpha=alpha, max_iter=max_iter, tol=tol
    )
    return scores, metadata
