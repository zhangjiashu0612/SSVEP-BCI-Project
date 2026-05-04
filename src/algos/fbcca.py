"""Filter-Bank Canonical Correlation Analysis (Chen et al. 2015).

Subbands: m * 8 - 90 Hz with m = 1..M. Chebyshev I order 6 IIR per band.
Per-target score = sum_n w(n) * r_n^2 where r_n = max canon corr in band n
and w(n) = n^(-a) + b with a=1.25, b=0.25.
"""
from __future__ import annotations

import numpy as np

from src.processing.filters import apply_sos, cheby1_bandpass_sos

from .base import Classifier, reference_signals
from .cca import _max_canon_corr


class FBCCA(Classifier):
    name = "fbcca"
    requires_training = False

    def __init__(self, freqs, fs, n_subbands: int = 7, low_cut: float = 8.0,
                 high_cut: float = 90.0, cheby_order: int = 6, cheby_rp: float = 0.5,
                 weight_a: float = 1.25, weight_b: float = 0.25, n_harmonics: int = 5,
                 phases=None):
        super().__init__(freqs, fs)
        self.n_subbands = n_subbands
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.cheby_order = cheby_order
        self.cheby_rp = cheby_rp
        self.weight_a = weight_a
        self.weight_b = weight_b
        self.n_harmonics = n_harmonics
        self.phases = list(phases) if phases is not None else None
        self._sos_bank = self._build_bank()
        self._weights = np.array(
            [(n ** (-weight_a)) + weight_b for n in range(1, n_subbands + 1)]
        )

    def _build_bank(self) -> list:
        return [
            cheby1_bandpass_sos(m * self.low_cut, self.high_cut, self.fs,
                                order=self.cheby_order, rp=self.cheby_rp)
            for m in range(1, self.n_subbands + 1)
        ]

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
            band_scores = np.zeros((self.n_subbands, len(self.freqs)))
            for m, sos in enumerate(self._sos_bank):
                xf = apply_sos(sos, x, axis=-1)
                for k, r in enumerate(refs):
                    rho = _max_canon_corr(xf, r)
                    band_scores[m, k] = rho ** 2
            combined = (self._weights[:, None] * band_scores).sum(axis=0)
            out[i] = int(np.argmax(combined))
        return out
