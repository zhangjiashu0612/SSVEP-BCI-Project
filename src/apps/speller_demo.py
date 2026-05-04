"""SSVEP speller demo entry point.

CLI:
  # 30 s headless mock — types "我想要喝水" by scripting a target schedule
  python -m src.apps.speller_demo --source mock --no-stim --duration 30

  # full PsychoPy UI
  python -m src.apps.speller_demo --source mock --duration 60

  # with the Cyton hardware
  python -m src.apps.speller_demo --source cyton --algo fbcca

The mock path runs without a display by default. To validate state-machine
progression headlessly, the mock source is given a deterministic schedule:
  t=2s  → look at letter 'w' (idx 22)
  t=6s  → look at candidate slot for '我'
  t=10s → look at candidate slot for '想要'
  t=14s → look at candidate slot for '喝水'
which produces the textbuffer "我想要喝水".
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src.acquisition import make_source
from src.algos import make_classifier
from src.processing.pipeline import SSVEPPipeline
from src.speller.layout import LETTERS, N_LETTERS, freq_grid
from src.speller.lm import LanguageModel
from src.speller.state import Layer, SpellerState
from src.utils.config import load_config, project_root


def _build_table(cfg: dict):
    g = cfg["speller"]["freq_grid"]
    return freq_grid(low_hz=g["low_hz"], step_hz=g["step_hz"],
                     n_targets=g["n_targets"], n_candidates=g["n_candidates"])


REST_IDX = -1  # MockSource convention: pure noise, breaks pipeline confidence


def _build_default_schedule(state: SpellerState) -> list[tuple[float, int]]:
    """Scripted schedule that types 我想要喝水 if the LM has those candidates.

    Each commit is followed by a brief noise rest so consecutive same-slot
    picks are seen as distinct trials by the pipeline.
    """
    sched: list[tuple[float, int]] = []
    DWELL = 5.5   # gaze duration; needs window_s + vote_window*step_ms
    REST = 3.0    # noise rest; needs to fully wash the prior signal out of the window

    def commit(t: float, idx: int) -> float:
        sched.append((t, idx))
        sched.append((t + DWELL, REST_IDX))
        return t + DWELL + REST

    t = 1.5
    t = commit(t, LETTERS.index("w"))
    cand_after_w = state.lm.predict_char("w", state.n_candidate_slots)
    if "我" not in cand_after_w:
        return sched
    t = commit(t, N_LETTERS + cand_after_w.index("我"))
    cand_after_我 = state.lm.predict_word("我", state.n_candidate_slots)
    if "想要" not in cand_after_我:
        return sched
    t = commit(t, N_LETTERS + cand_after_我.index("想要"))
    cand_after_想要 = state.lm.predict_continuation(
        "想要", state.n_candidate_slots)
    if "喝水" not in cand_after_想要:
        return sched
    t = commit(t, N_LETTERS + cand_after_想要.index("喝水"))
    return sched


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["mock", "cyton"], default="mock")
    p.add_argument("--algo", choices=["fbcca", "cca", "trca", "psda"], default="fbcca")
    p.add_argument("--config", default=str(project_root() / "config" / "speller.yaml"))
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--no-stim", action="store_true")
    p.add_argument("--seed-schedule", action="store_true",
                   help="(mock only) auto-script a typing sequence")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    fs = cfg["acquisition"]["fs"]
    channels = cfg["acquisition"]["channels"]
    table = _build_table(cfg)

    lm = LanguageModel.from_resources(
        project_root() / cfg["speller"]["lm_resources_dir"]
    )
    state = SpellerState(lm=lm,
                         n_candidate_slots=cfg["speller"]["freq_grid"]["n_candidates"])

    source = make_source(args.source, cfg, freqs=table.freqs.tolist())
    if args.source == "mock":
        # mock source needs the right freq+phase set so the synth matches the targets
        source.freqs = table.freqs
        if args.seed_schedule or args.no_stim:
            source.set_schedule(_build_default_schedule(state))

    clf = make_classifier(args.algo, table.freqs.tolist(), fs, cfg,
                          phases=table.phases.tolist())
    pipeline = SSVEPPipeline(
        classifier=clf, fs=fs, n_channels=len(channels),
        window_s=cfg["processing"]["window_s"],
        step_ms=cfg["processing"]["step_ms"],
        ring_buffer_s=cfg["processing"]["ring_buffer_s"],
        vote_window=cfg["processing"]["vote_window"],
        bandpass=cfg["processing"]["bandpass"],
        notch_hz=cfg["processing"]["notch_hz"],
        filter_order=cfg["processing"]["filter_order"],
    )

    stim = None
    if not args.no_stim:
        try:
            from src.stimulus.speller_stim import SpellerStimulus
            stim = SpellerStimulus(
                freqs=table.freqs.tolist(),
                phases=table.phases.tolist(),
                letter_layout=cfg["stimulus"]["letter_layout"],
                n_candidates=cfg["speller"]["freq_grid"]["n_candidates"],
                monitor_hz=cfg["stimulus"]["monitor_hz"],
                cell_size_px=cfg["stimulus"]["size_px"],
                candidate_size_px=tuple(cfg["stimulus"]["candidate_slot_size_px"]),
                fullscreen=cfg["stimulus"]["fullscreen"],
                window_title=cfg["speller"]["window_title"],
            )
        except Exception as e:
            print(f"[speller] PsychoPy disabled: {e}")
            stim = None

    # commit gating: consume at most one commit per "gaze episode". Reset when
    # the pipeline's vote_window can't agree (confirmed_idx is None) for a
    # stretch — that's the equivalent of the user looking away.
    last_confirmed: int | None = None
    no_confirm_streak = 0
    RESET_AFTER = 3  # consecutive None confirmations before re-arming

    def on_pred(p):
        nonlocal last_confirmed, no_confirm_streak
        if p.confirmed_idx is None:
            no_confirm_streak += 1
            if no_confirm_streak >= RESET_AFTER:
                last_confirmed = None
            if stim is not None:
                stim.set_prediction(p.raw_idx)
            return
        no_confirm_streak = 0
        if p.confirmed_idx == last_confirmed:
            if stim is not None:
                stim.set_prediction(p.confirmed_idx)
            return
        last_confirmed = p.confirmed_idx
        tr = state.on_target(p.confirmed_idx)
        print(f"[state] layer={tr.layer.value:6s} action={tr.last_action:30s} "
              f"buffer={tr.text_buffer!r}  candidates={tr.candidates}")
        if stim is not None:
            stim.set_prediction(p.confirmed_idx)
            stim.set_candidates(tr.candidates)
            stim.set_text_buffer(tr.text_buffer)
            stim.emit_marker(f"speller:{tr.last_action}|buf={tr.text_buffer}")

    pipeline.on_prediction(on_pred)
    source.start()
    try:
        pipeline.start(chunk_fn=lambda: source.read_chunk())
        if stim is not None:
            stim.run(duration_s=args.duration)
        else:
            time.sleep(args.duration)
    finally:
        pipeline.stop()
        source.stop()
        if stim is not None:
            stim.close()

    print(f"\n[speller] final buffer: {state.text_buffer!r}")


if __name__ == "__main__":
    main()
