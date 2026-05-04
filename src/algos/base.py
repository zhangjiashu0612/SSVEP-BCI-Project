"""Unified classifier interface used by all SSVEP algorithms."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class Classifier(ABC):
    """Common interface.

    X has shape (n_trials, n_channels, n_samples). y has shape (n_trials,)
    with integer class labels matching the index of `freqs`.

    Training-free models (PSDA/CCA/FBCCA) implement `fit` as a no-op and
    only require `freqs` and sampling rate at construction.
    """

    name: str = "base"
    requires_training: bool = False

    def __init__(self, freqs: Sequence[float], fs: float):
        self.freqs = np.asarray(freqs, dtype=float)
        self.fs = float(fs)

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Classifier":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


def reference_signals(freqs: Sequence[float], fs: float, n_samples: int,
                      n_harmonics: int = 5) -> list[np.ndarray]:
    """Sin/cos reference banks per target frequency. Each entry is (2*n_harmonics, n_samples)."""
    t = np.arange(n_samples) / fs
    refs = []
    for f in freqs:
        rows = []
        for h in range(1, n_harmonics + 1):
            rows.append(np.sin(2 * np.pi * h * f * t))
            rows.append(np.cos(2 * np.pi * h * f * t))
        refs.append(np.vstack(rows))
    return refs
