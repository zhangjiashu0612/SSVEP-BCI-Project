"""Canonical Correlation Analysis between EEG and sin/cos reference banks."""
from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import CCA as _SkCCA

from .base import Classifier, reference_signals


def _max_canon_corr(x: np.ndarray, ref: np.ndarray, n_components: int = 1) -> float:
    """x: (channels, samples), ref: (2*harmonics, samples)."""
    n = min(x.shape[0], ref.shape[0], n_components)
    cca = _SkCCA(n_components=n, max_iter=500)
    try:
        a, b = cca.fit_transform(x.T, ref.T)
    except Exception:
        return 0.0
    rs = []
    for k in range(n):
        ak, bk = a[:, k], b[:, k]
        sa, sb = ak.std(), bk.std()
        if sa < 1e-12 or sb < 1e-12:
            rs.append(0.0)
        else:
            rs.append(float(np.corrcoef(ak, bk)[0, 1]))
    return max(rs) if rs else 0.0


class CCA(Classifier):
    name = "cca"
    requires_training = False

    def __init__(self, freqs, fs, n_harmonics: int = 5, phases=None):
        super().__init__(freqs, fs)
        self.n_harmonics = n_harmonics
        self.phases = list(phases) if phases is not None else None

    def fit(self, X, y):
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            X = X[None]
        n_samples = X.shape[-1]
        refs = reference_signals(self.freqs, self.fs, n_samples,
                                 self.n_harmonics, phases=self.phases)
        out = np.empty(len(X), dtype=int)
        for i, x in enumerate(X):
            scores = [_max_canon_corr(x, r) for r in refs]
            out[i] = int(np.argmax(scores))
        return out
