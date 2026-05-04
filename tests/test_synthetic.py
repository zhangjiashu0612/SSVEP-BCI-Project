"""CCA must correctly pick the injected target from a synthetic SSVEP.

Pipeline:
  - 8 channels, fs=250 Hz, 4 s
  - inject 12 Hz sinusoid (+ harmonics) with phase jitter per channel
  - target set: [8, 10, 12, 15] Hz
  - SNR ≥ 0 dB → CCA must pick index 2 (== 12 Hz)
"""
from __future__ import annotations

import numpy as np
import pytest

from src.algos.cca import CCA
from src.algos.fbcca import FBCCA
from src.algos.psda import PSDA


FS = 250
DURATION = 4.0
N_CHANNELS = 8
FREQS = [8.0, 10.0, 12.0, 15.0]
TARGET_IDX = 2


def _synth(target_hz: float, snr_db: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(FS * DURATION)) / FS
    sig = np.zeros((N_CHANNELS, t.size), dtype=np.float64)
    for h, amp in [(1, 1.0), (2, 0.5), (3, 0.3)]:
        phase = rng.uniform(0, 2 * np.pi, size=N_CHANNELS)[:, None]
        sig += amp * np.sin(2 * np.pi * h * target_hz * t + phase)
    sig_p = np.mean(sig ** 2)
    snr = 10 ** (snr_db / 10)
    noise_p = sig_p / snr
    noise = rng.normal(0, np.sqrt(noise_p), size=sig.shape)
    return sig + noise


@pytest.mark.parametrize("snr_db", [0, 3, 6])
def test_cca_identifies_12hz(snr_db):
    x = _synth(12.0, snr_db=snr_db, seed=snr_db)
    pred = CCA(FREQS, FS).predict(x[None])[0]
    assert pred == TARGET_IDX, f"CCA picked {FREQS[pred]} at SNR {snr_db} dB"


def test_fbcca_identifies_12hz_low_snr():
    x = _synth(12.0, snr_db=-3, seed=11)
    pred = FBCCA(FREQS, FS).predict(x[None])[0]
    assert pred == TARGET_IDX


def test_psda_identifies_12hz():
    x = _synth(12.0, snr_db=6, seed=7)
    pred = PSDA(FREQS, FS).predict(x[None])[0]
    assert pred == TARGET_IDX
