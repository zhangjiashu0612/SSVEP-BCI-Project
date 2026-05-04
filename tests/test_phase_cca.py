"""Phase-aware reference signals: smoke tests on the API plumbing.

Standard CCA with sin+cos pairs at each harmonic spans a phase-invariant
2D subspace, so CCA cannot discriminate same-frequency-different-phase
targets — and we don't try to. These tests only verify:

  1. Passing `phases=None` produces phase-zero behavior (regression).
  2. Passing `phases` doesn't break per-frequency selection.
  3. Reference banks include the phase shift in the harmonics.

Phase plumbing exists for a future TRCA-on-JFPM extension, where the spatial
filter + template carry the phase information.
"""
from __future__ import annotations

import numpy as np

from src.algos.base import reference_signals
from src.algos.cca import CCA
from src.algos.fbcca import FBCCA


FS = 250
N = int(FS * 2.0)
N_CHANNELS = 8


def _synth(f: float, snr_db: float = 6.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    sig = np.zeros((N_CHANNELS, N))
    for h, amp in [(1, 1.0), (2, 0.5), (3, 0.3)]:
        phase = rng.uniform(0, 2 * np.pi, size=N_CHANNELS)[:, None]
        sig += amp * np.sin(2 * np.pi * h * f * t + phase)
    sig_p = np.mean(sig ** 2)
    noise = rng.normal(0, np.sqrt(sig_p / (10 ** (snr_db / 10))), size=sig.shape)
    return sig + noise


def test_reference_signals_phase_shift_appears():
    refs0 = reference_signals([10.0], FS, N, n_harmonics=2, phases=[0.0])
    refsp = reference_signals([10.0], FS, N, n_harmonics=2, phases=[np.pi / 2])
    # sin(2πft + π/2) == cos(2πft); these reference banks must differ.
    assert not np.allclose(refs0[0], refsp[0])


def test_phases_default_matches_zero_phases():
    refs_none = reference_signals([10.0, 12.0], FS, N, n_harmonics=3, phases=None)
    refs_zero = reference_signals([10.0, 12.0], FS, N, n_harmonics=3,
                                  phases=[0.0, 0.0])
    for a, b in zip(refs_none, refs_zero):
        assert np.allclose(a, b)


def test_cca_with_phases_still_picks_correct_freq():
    """Phase-aware CCA must still discriminate distinct frequencies."""
    freqs = [8.0, 10.0, 12.0, 15.0]
    phases = [0.0, 0.0, np.pi / 4, 0.0]  # arbitrary — should not affect freq pick
    for target_idx, f in enumerate(freqs):
        x = _synth(f, snr_db=6, seed=target_idx)
        clf = CCA(freqs, FS, phases=phases)
        assert clf.predict(x[None])[0] == target_idx


def test_fbcca_phase_param_is_optional():
    """No phases passed = original behaviour."""
    freqs = [8.0, 10.0, 12.0, 15.0]
    x = _synth(12.0, snr_db=6, seed=11)
    p1 = FBCCA(freqs, FS).predict(x[None])[0]
    p2 = FBCCA(freqs, FS, phases=[0, 0, 0, 0]).predict(x[None])[0]
    assert p1 == p2 == 2


def test_dense_freq_grid_separability_at_2s():
    """32-target Wang-style grid: FBCCA picks the right freq at SNR=8 in a 2 s window."""
    grid = [8.0 + 0.2 * i for i in range(32)]
    target_idx = 15
    x = _synth(grid[target_idx], snr_db=8, seed=target_idx)
    pred = FBCCA(grid, FS).predict(x[None])[0]
    # At 0.2 Hz spacing some neighbour confusion is expected; require within ±1.
    assert abs(pred - target_idx) <= 1, f"FBCCA picked {grid[pred]} for target {grid[target_idx]}"
