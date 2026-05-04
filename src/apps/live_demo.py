"""Real-time SSVEP demo entry point.

CLI:
  python -m src.apps.live_demo --source mock --algo fbcca
  python -m src.apps.live_demo --source cyton --algo trca --trca-model models/trca_wang2016.pkl
  python -m src.apps.live_demo --source mock --algo fbcca --no-stim   # headless

The default flow:
  acquisition source -> LSL outlet -> pipeline (LSL inlet) -> stim highlight
For development we allow `--direct` to skip LSL and pull chunks straight from
the source — handy when you don't have a working LSL background process.
"""
from __future__ import annotations

import argparse
import pickle
import threading
import time
from pathlib import Path

import numpy as np

from src.acquisition import make_source
from src.algos import make_classifier
from src.processing.pipeline import SSVEPPipeline, lsl_chunk_fn
from src.utils.config import load_config


def _load_classifier(algo: str, freqs, fs, cfg: dict, trca_model: Path | None):
    if algo == "trca" and trca_model is not None:
        with open(trca_model, "rb") as f:
            payload = pickle.load(f)
        print(f"[live] loaded TRCA model from {trca_model} "
              f"(subject={payload.get('subject')}, dataset={payload.get('dataset')})")
        return payload["model"]
    return make_classifier(algo, freqs, fs, cfg)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["cyton", "mock", "replay"], default="mock")
    parser.add_argument("--algo", choices=["fbcca", "trca", "cca", "psda"], default="fbcca")
    parser.add_argument("--trca-model", default="models/trca_wang2016.pkl")
    parser.add_argument("--config", default=None)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--no-stim", action="store_true")
    parser.add_argument("--direct", action="store_true",
                        help="skip LSL inlet; pipeline pulls from source.read_chunk()")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    fs = cfg["acquisition"]["fs"]
    freqs = cfg["stimulus"]["freqs_hz"]
    channels = cfg["acquisition"]["channels"]

    if args.source == "replay":
        raise NotImplementedError("replay source not implemented yet — use --source mock")
    source = make_source(args.source, cfg, freqs=freqs)
    clf = _load_classifier(
        args.algo, freqs, fs, cfg,
        Path(args.trca_model) if args.algo == "trca" else None,
    )
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
            from src.stimulus.ssvep_stim import SSVEPStimulus
            stim = SSVEPStimulus(
                freqs=freqs, phases=cfg["stimulus"]["phases_rad"],
                monitor_hz=cfg["stimulus"]["monitor_hz"],
                size_px=cfg["stimulus"]["size_px"],
                fullscreen=cfg["stimulus"]["fullscreen"],
            )
        except Exception as e:
            print(f"[live] PsychoPy stim disabled: {e}")
            stim = None

    def on_pred(p):
        marker = (f"freq={p.score_freq_hz:.2f}Hz "
                  f"raw={p.raw_idx} confirmed={p.confirmed_idx} "
                  f"lat={p.latency_ms:.1f}ms")
        print(f"[pred] {marker}")
        if stim is not None and p.confirmed_idx is not None:
            stim.set_prediction(p.confirmed_idx)
            stim.emit_marker(f"predict:{p.confirmed_idx}:{p.score_freq_hz:.2f}")
    pipeline.on_prediction(on_pred)

    chunk_fn = None
    close_fn = None
    source.start()
    try:
        if args.direct or args.source == "mock":
            chunk_fn = lambda: source.read_chunk()
        else:
            time.sleep(0.3)  # let outlet settle
            chunk_fn, close_fn = lsl_chunk_fn(stream_name="ssvep_eeg",
                                              n_channels=len(channels))
        pipeline.start(chunk_fn)

        if stim is not None:
            stim.run(duration_s=args.duration)
        else:
            time.sleep(args.duration)
    finally:
        pipeline.stop()
        if close_fn is not None:
            try:
                close_fn()
            except Exception:
                pass
        source.stop()
        if stim is not None:
            stim.close()


if __name__ == "__main__":
    main()
