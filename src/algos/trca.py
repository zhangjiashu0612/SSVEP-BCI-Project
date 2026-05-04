"""Task-Related Component Analysis (Nakanishi et al. 2018).

Standard formulation: per class c with trials X_c (K, C, T), find spatial
filter w_c maximizing inter-trial covariance ratio
    w^T S w / w^T Q w
S = sum_{k != l} X_k X_l^T,  Q = sum_k X_k X_k^T.
At test time score class c by Pearson r between w_c-projected test trial
and the w_c-projected class template (mean across training trials).

Filter-bank extension: same as FBCCA with subband m = 1..M and weights
w(n) = n^(-1.25) + 0.25.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eig

from src.processing.filters import apply_sos, cheby1_bandpass_sos

from .base import Classifier


def _trca_filter(X_c: np.ndarray) -> np.ndarray:
    """X_c: (K, C, T). Return spatial filter w of shape (C,)."""
    K, C, T = X_c.shape
    Xc = X_c - X_c.mean(axis=2, keepdims=True)
    S = np.zeros((C, C))
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            S += Xc[i] @ Xc[j].T
    Xcat = Xc.reshape(C, K * T)
    Q = Xcat @ Xcat.T
    Q += 1e-6 * np.trace(Q) / C * np.eye(C)
    vals, vecs = eig(S, Q)
    vals = np.real(vals)
    return np.real(vecs[:, np.argmax(vals)])


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.mean(a * b) / (sa * sb))


class TRCA(Classifier):
    name = "trca"
    requires_training = True

    def __init__(self, freqs, fs, filterbank: bool = True, n_subbands: int = 7,
                 low_cut: float = 8.0, high_cut: float = 90.0,
                 cheby_order: int = 6, cheby_rp: float = 0.5,
                 weight_a: float = 1.25, weight_b: float = 0.25):
        super().__init__(freqs, fs)
        self.filterbank = filterbank
        self.n_subbands = n_subbands if filterbank else 1
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.cheby_order = cheby_order
        self.cheby_rp = cheby_rp
        self._sos_bank = self._build_bank() if filterbank else None
        self._weights = np.array(
            [(n ** (-weight_a)) + weight_b for n in range(1, self.n_subbands + 1)]
        )
        self.filters_ = None    # list[ndarray] indexed by (band, class) -> (C,)
        self.templates_ = None  # list[ndarray] indexed by (band, class) -> (T,)

    def _build_bank(self):
        return [
            cheby1_bandpass_sos(m * self.low_cut, self.high_cut, self.fs,
                                order=self.cheby_order, rp=self.cheby_rp)
            for m in range(1, self.n_subbands + 1)
        ]

    def _bands(self, X: np.ndarray) -> list[np.ndarray]:
        """Return list of length n_subbands of arrays shaped like X."""
        if not self.filterbank:
            return [X]
        return [apply_sos(sos, X, axis=-1) for sos in self._sos_bank]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TRCA":
        classes = np.unique(y)
        if not np.array_equal(classes, np.arange(len(self.freqs))):
            raise ValueError("y must contain class indices 0..n_classes-1")
        bands = self._bands(X)
        self.filters_ = []
        self.templates_ = []
        for Xb in bands:
            fb_filters = []
            fb_templates = []
            for c in classes:
                Xc = Xb[y == c]
                w = _trca_filter(Xc)
                fb_filters.append(w)
                template = (w @ Xc.mean(axis=0))
                fb_templates.append(template)
            self.filters_.append(fb_filters)
            self.templates_.append(fb_templates)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.filters_ is None:
            raise RuntimeError("TRCA not fitted")
        if X.ndim == 2:
            X = X[None]
        bands = self._bands(X)
        out = np.empty(len(X), dtype=int)
        n_classes = len(self.freqs)
        for i in range(len(X)):
            scores = np.zeros(n_classes)
            for m, Xb in enumerate(bands):
                for c in range(n_classes):
                    proj = self.filters_[m][c] @ Xb[i]
                    rho = _corr(proj, self.templates_[m][c])
                    scores[c] += self._weights[m] * (rho ** 2)
            out[i] = int(np.argmax(scores))
        return out
