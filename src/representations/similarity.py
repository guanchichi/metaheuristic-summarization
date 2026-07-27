from typing import Union
import numpy as np


def cosine_similarity_matrix(X: Union[np.ndarray, "scipy.sparse.spmatrix"]) -> np.ndarray:
    """Return the declared sklearn cosine implementation or fail loudly.

    A broad fallback used to turn dependency, shape, and non-finite-input errors
    into a different NumPy implementation. That made the effective method depend
    on the machine and could let invalid matrices continue through a formal run.
    """

    from sklearn.metrics.pairwise import cosine_similarity

    return cosine_similarity(X)
