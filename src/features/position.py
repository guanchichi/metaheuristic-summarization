from typing import List
import math


# ---------- original API (kept for backward compat) ----------

def position_scores(sentences: List[str]) -> List[float]:
    n = len(sentences)
    if n == 0:
        return []
    # Higher score for earlier sentences (descending linear)
    return [1.0 - (i / max(1, n - 1)) if n > 1 else 1.0 for i in range(n)]


# ---------- improved version ----------

def position_scores_v2(
    sentences: List[str],
    method: str = "inverse",
    decay: float = 0.1,
) -> List[float]:
    """Position scoring with configurable decay functions.

    Parameters
    ----------
    method : str
        ``"linear"``  – original linear decay ``1 - i/(n-1)``
        ``"inverse"`` – ``1 / (1 + i)`` (stronger lead bias)
        ``"exponential"`` – ``exp(-decay * i)`` (configurable)
    decay : float
        Decay rate for the exponential method.
    """
    normalized_method = (method or "").strip().lower()
    if normalized_method not in {"linear", "inverse", "exponential"}:
        raise ValueError(f"unknown position scoring method: {method!r}")
    if not math.isfinite(decay) or decay < 0:
        raise ValueError("position exponential decay must be finite and non-negative")

    n = len(sentences)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    raw: List[float] = []
    for i in range(n):
        if normalized_method == "linear":
            raw.append(1.0 - (i / (n - 1)))
        elif normalized_method == "inverse":
            raw.append(1.0 / (1.0 + i))
        elif normalized_method == "exponential":
            raw.append(math.exp(-decay * i))

    # normalize to [0, 1]
    mx = max(raw)
    return [s / mx for s in raw] if mx > 0 else raw
