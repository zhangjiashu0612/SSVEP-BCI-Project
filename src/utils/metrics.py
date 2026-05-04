"""Evaluation metrics for SSVEP BCI."""
from __future__ import annotations

import math


def wolpaw_itr(accuracy: float, n_classes: int, trial_duration_s: float) -> float:
    """Information Transfer Rate in bits/min (Wolpaw et al. 1998).

    P=1 → log2(N); P<1/N clipped to 0; otherwise the standard formula.
    """
    if n_classes < 2 or trial_duration_s <= 0:
        return 0.0
    p = max(0.0, min(1.0, accuracy))
    if p <= 1.0 / n_classes:
        return 0.0
    if p == 1.0:
        bits = math.log2(n_classes)
    else:
        bits = (
            math.log2(n_classes)
            + p * math.log2(p)
            + (1 - p) * math.log2((1 - p) / (n_classes - 1))
        )
    return bits * 60.0 / trial_duration_s
