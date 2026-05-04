"""End-to-end mock speller test.

Drives the MockSource on a scripted target schedule (look at 'w', then a
candidate slot), runs through the full SSVEPPipeline + SpellerState, and
checks that at least one Chinese character ends up in the text buffer.
"""
from __future__ import annotations

import time

import pytest

from src.acquisition.mock import MockSource
from src.algos.fbcca import FBCCA
from src.processing.pipeline import SSVEPPipeline
from src.speller.layout import LETTERS, N_LETTERS, freq_grid
from src.speller.lm import LanguageModel
from src.speller.state import Layer, SpellerState


@pytest.mark.timeout(60)
def test_speller_typing_w_then_我():
    table = freq_grid(low_hz=8.0, step_hz=0.2, n_targets=32, n_candidates=6)
    fs = 250
    channels = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]

    src = MockSource(freqs=table.freqs.tolist(), channels=channels, fs=fs,
                     snr_db=15.0, seed=7, stream_name="speller_test_eeg")
    src.freqs = table.freqs

    lm = LanguageModel.from_resources(None)
    state = SpellerState(lm=lm, n_candidate_slots=6)

    # plan: at t=2 look at 'w', at t=7 look at slot for '我'
    cand_after_w = lm.predict_char("w", 6)
    assert "我" in cand_after_w
    slot_我 = cand_after_w.index("我")
    src.set_schedule([
        (2.0, LETTERS.index("w")),
        (7.0, N_LETTERS + slot_我),
    ])

    clf = FBCCA(table.freqs.tolist(), fs, phases=table.phases.tolist(),
                n_harmonics=3)
    pipe = SSVEPPipeline(classifier=clf, fs=fs, n_channels=len(channels),
                         window_s=2.5, step_ms=300, ring_buffer_s=8.0,
                         vote_window=3)

    last_confirmed = {"v": None}

    def on_pred(p):
        if p.confirmed_idx is None:
            return
        if p.confirmed_idx == last_confirmed["v"]:
            return
        last_confirmed["v"] = p.confirmed_idx
        state.on_target(p.confirmed_idx)

    pipe.on_prediction(on_pred)

    src.start()
    try:
        pipe.start(chunk_fn=lambda: src.read_chunk())
        # need ≥ window_s + schedule end + voting wait time
        time.sleep(11.0)
    finally:
        pipe.stop()
        src.stop()

    assert state.text_buffer == "我", (
        f"expected text_buffer to advance to '我', got {state.text_buffer!r}; "
        f"history={state.history}"
    )
    assert state.layer is Layer.CHAR
