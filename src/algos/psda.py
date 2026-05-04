"""Power Spectral Density Analysis: pick the target whose harmonics have most power."""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

from .base import Classifier


class PSDA(Classifier):
    name = "psda"
    requires_training = False

    def __init__(self, freqs, fs, n_harmonics: int = 3, freq_resolution_hz: float = 0.1):
        super().__init__(freqs, fs)
        self.n_harmonics = n_harmonics
        self.freq_resolution_hz = freq_resolution_hz

    def fit(self, X, y):
        return self

    def _trial_score(self, x: np.ndarray) -> np.ndarray:
        nperseg = min(x.shape[-1], int(self.fs / self.freq_resolution_hz))
        nperseg = max(64, nperseg)
        f, pxx = welch(x, fs=self.fs, nperseg=nperseg, axis=-1)
        pxx_mean = pxx.mean(axis=0)  # average across channels
        scores = np.zeros(len(self.freqs))
        for i, f0 in enumerate(self.freqs):
            for h in range(1, self.n_harmonics + 1):
                idx = int(np.argmin(np.abs(f - h * f0)))
                lo = max(0, idx - 1)
                hi = min(len(f), idx + 2)
                scores[i] += pxx_mean[lo:hi].max()
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            X = X[None]
        return np.array([np.argmax(self._trial_score(x)) for x in X])
