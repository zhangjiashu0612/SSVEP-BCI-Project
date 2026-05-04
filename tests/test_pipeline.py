"""Run the real-time pipeline against the mock source for ~5 s and verify
that it produces predictions and at least one of them matches the injected
target.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from src.acquisition.mock import MockSource
from src.algos.fbcca import FBCCA
from src.processing.pipeline import SSVEPPipeline
from src.utils.config import load_config


FREQS = [7.5, 8.57142857, 10.0, 12.0]
CHANNELS = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]
FS = 250


@pytest.mark.timeout(30)
def test_pipeline_runs_with_mock_source():
    src = MockSource(freqs=FREQS, channels=CHANNELS, fs=FS, snr_db=8.0, seed=3,
                     stream_name="ssvep_eeg_test")
    src.set_target(2)  # 10 Hz

    clf = FBCCA(FREQS, FS)
    pipe = SSVEPPipeline(classifier=clf, fs=FS, n_channels=len(CHANNELS),
                         window_s=2.0, step_ms=200, ring_buffer_s=4.0,
                         vote_window=3)
    src.start()
    try:
        pipe.start(chunk_fn=lambda: src.read_chunk())
        time.sleep(5.0)
        pipe.stop()
    finally:
        src.stop()

    preds = pipe.predictions
    assert len(preds) >= 5, f"expected >= 5 preds, got {len(preds)}"

    # warm-up: skip first prediction (buffer not full)
    raw_idxs = [p.raw_idx for p in preds[1:]]
    assert raw_idxs.count(2) >= len(raw_idxs) // 2, \
        f"FBCCA should pick 10 Hz most of the time, got {raw_idxs}"

    # at least one confirmation event after voting kicks in
    confirmed = [p.confirmed_idx for p in preds if p.confirmed_idx is not None]
    assert any(c == 2 for c in confirmed), \
        f"expected at least one confirmed=2, got {confirmed}"


def test_pipeline_buffer_underflow_returns_none():
    """Sanity: ring buffer must wait until enough samples accumulate."""
    from src.processing.pipeline import RingBuffer

    rb = RingBuffer(n_channels=4, capacity_samples=100)
    assert rb.latest(50) is None
    rb.push(np.ones((4, 30), dtype=np.float32))
    assert rb.latest(50) is None
    rb.push(np.ones((4, 30), dtype=np.float32))
    win = rb.latest(50)
    assert win is not None and win.shape == (4, 50)
