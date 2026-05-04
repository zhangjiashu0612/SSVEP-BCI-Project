"""Offline benchmark across (algorithm × dataset × subject × window length).

For each subject we use leave-one-block-out CV. PSDA/CCA/FBCCA are training-free
so they ignore the train fold; TRCA is fitted on it. For the Wang2016 dataset
we also pickle one subject's TRCA model to models/trca_wang2016.pkl so the
live demo can load it.

This script is robust to MOABB version differences: it expects
get_data() returning {subject: {session: {run: Raw}}} or the older
{subject: {session: {run: (Raw, events)}}}; it normalizes both.
"""
from __future__ import annotations

import argparse
import pickle
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

from src.algos import make_classifier
from src.algos.trca import TRCA
from src.utils.config import load_config, project_root
from src.utils.metrics import wolpaw_itr
from src.utils.plots import accuracy_vs_window, confusion_matrix_plot, itr_bar

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---- dataset helpers --------------------------------------------------------

def _load_dataset(name: str):
    """Return a moabb dataset instance and metadata dict."""
    from moabb.datasets import Nakanishi2015, Wang2016

    if name == "Wang2016":
        return Wang2016(), {"epoch_window_s": 5.0}
    if name == "Nakanishi2015":
        return Nakanishi2015(), {"epoch_window_s": 4.0}
    raise ValueError(f"unknown dataset: {name}")


def _get_freqs(dataset) -> list[float]:
    """MOABB datasets carry events keyed by frequency strings."""
    events = dataset.event_id
    freqs = []
    for k in events.keys():
        try:
            freqs.append(float(k))
        except ValueError:
            try:
                freqs.append(float(k.split("Hz")[0].strip()))
            except Exception:
                pass
    if not freqs:
        raise RuntimeError(f"could not extract freqs from {dataset}")
    return sorted(freqs)


def _epoch_subject(dataset, subject: int, fs_target: int | None = None,
                   tmin: float = 0.14, tmax: float | None = None,
                   bandpass=(6.0, 90.0)):
    """Return X (n_trials, n_ch, n_samples), y (n_trials,), labels-as-freqs.

    tmin=0.14 matches Wang2016 stim onset latency convention (140 ms after cue).
    """
    import mne

    raws = dataset.get_data(subjects=[subject])[subject]
    epochs_list = []
    labels_list = []
    event_id = dataset.event_id
    inv_event_id = {v: k for k, v in event_id.items()}
    freqs_sorted = sorted({float(k.split("Hz")[0].strip() if "Hz" in k else k)
                           for k in event_id.keys()})
    label_to_idx = {f: i for i, f in enumerate(freqs_sorted)}

    for session, runs in raws.items():
        for run, item in runs.items():
            raw = item if isinstance(item, mne.io.BaseRaw) else item[0]
            raw = raw.copy()
            if fs_target and int(round(raw.info["sfreq"])) != fs_target:
                raw.resample(fs_target, npad="auto")
            raw.filter(bandpass[0], bandpass[1], verbose=False)
            picks = mne.pick_types(raw.info, eeg=True)
            events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
            ep_tmax = tmax if tmax is not None else (
                tmin + 4.9 if "Wang" in dataset.__class__.__name__ else tmin + 3.9
            )
            ep = mne.Epochs(raw, events, event_id=event_id, tmin=tmin,
                            tmax=ep_tmax, baseline=None, picks=picks,
                            preload=True, verbose=False)
            X = ep.get_data()  # (n, c, t)
            y_codes = ep.events[:, -1]
            y = np.array([
                label_to_idx[float(inv_event_id[c].split("Hz")[0].strip()
                                   if "Hz" in inv_event_id[c] else inv_event_id[c])]
                for c in y_codes
            ])
            epochs_list.append(X)
            labels_list.append(y)
    if not epochs_list:
        raise RuntimeError(f"no epochs for subject {subject}")
    X_all = np.concatenate(epochs_list, axis=0)
    y_all = np.concatenate(labels_list, axis=0)
    return X_all.astype(np.float32), y_all.astype(int), freqs_sorted, raws


def _block_ids(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Reconstruct block ids assuming each block contains one trial per class."""
    blocks = np.zeros_like(y)
    seen = np.zeros(n_classes, dtype=int)
    for i, cls in enumerate(y):
        blocks[i] = seen[cls]
        seen[cls] += 1
    return blocks


# ---- main loop --------------------------------------------------------------

def run_subject(dataset_name: str, subject: int, X: np.ndarray, y: np.ndarray,
                freqs: list[float], fs: float, windows_s: list[float],
                algos: list[str], cfg: dict, save_trca_path: Path | None = None) -> list[dict]:
    rows = []
    n_classes = len(freqs)
    blocks = _block_ids(y, n_classes)
    n_blocks = int(blocks.max() + 1)
    n_total = X.shape[-1]

    saved_trca = False
    for w_s in windows_s:
        n_win = min(int(round(w_s * fs)), n_total)
        Xw = X[..., :n_win]

        for algo in algos:
            clf_factory = lambda: make_classifier(algo, freqs, fs, cfg)
            preds_all = np.full(len(y), -1, dtype=int)
            latencies = []
            for b in range(n_blocks):
                test_mask = blocks == b
                train_mask = ~test_mask
                clf = clf_factory()
                if clf.requires_training:
                    clf.fit(Xw[train_mask], y[train_mask])
                t0 = time.perf_counter()
                p = clf.predict(Xw[test_mask])
                latencies.append((time.perf_counter() - t0) / max(test_mask.sum(), 1) * 1000)
                preds_all[test_mask] = p
                if (algo == "trca" and save_trca_path is not None and not saved_trca
                        and dataset_name == "Wang2016" and w_s == max(windows_s)
                        and b == 0):
                    save_trca_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(save_trca_path, "wb") as f:
                        pickle.dump({
                            "model": clf, "freqs": freqs, "fs": fs, "window_s": w_s,
                            "subject": subject, "channels": cfg["acquisition"]["channels"],
                            "dataset": dataset_name,
                        }, f)
                    saved_trca = True

            acc = float(np.mean(preds_all == y))
            itr = wolpaw_itr(acc, n_classes, w_s)
            rows.append({
                "dataset": dataset_name, "subject": subject, "algo": algo,
                "window_s": w_s, "accuracy": acc, "itr_bpm": itr,
                "latency_ms_per_trial": float(np.mean(latencies)),
                "n_trials": int(len(y)), "n_classes": n_classes,
            })
            print(f"  [{dataset_name} S{subject:02d} {algo:>5s} {w_s:.1f}s] "
                  f"acc={acc:.3f} ITR={itr:.1f} bpm  lat={np.mean(latencies):.1f} ms/trial")

            if algo == "trca" and w_s == max(windows_s):
                cm = confusion_matrix(y, preds_all, labels=list(range(n_classes)))
                out = project_root() / "results" / f"cm_{dataset_name}_S{subject:02d}_trca.png"
                confusion_matrix_plot(cm, [f"{f:.2f}" for f in freqs], out,
                                      f"{dataset_name} S{subject} TRCA {w_s:g}s")
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--subjects", nargs="+", type=int, default=None,
                        help="subset of subjects (default: dataset.subject_list)")
    parser.add_argument("--algos", nargs="+", default=None)
    parser.add_argument("--windows", nargs="+", type=float, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    cfg = load_config()
    bench = cfg["benchmark"]
    datasets = args.datasets or bench["datasets"]
    algos = args.algos or bench["algos"]
    windows_s = args.windows or bench["windows_s"]
    fs = cfg["acquisition"]["fs"]

    out_dir = project_root() / bench["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.out) if args.out else out_dir / "benchmark.csv"
    models_dir = project_root() / bench["models_dir"]
    trca_save_path = models_dir / "trca_wang2016.pkl"

    rows: list[dict] = []
    for ds_name in datasets:
        ds, meta = _load_dataset(ds_name)
        subjects = args.subjects or ds.subject_list
        for subject in subjects:
            print(f"[load] {ds_name} subject {subject}")
            try:
                X, y, freqs, _ = _epoch_subject(ds, subject, fs_target=fs)
            except Exception as e:
                print(f"  skip subject {subject}: {e}")
                continue
            rows.extend(run_subject(
                ds_name, subject, X, y, freqs, fs, windows_s, algos, cfg,
                save_trca_path=trca_save_path if ds_name == "Wang2016" else None,
            ))
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\n[done] wrote {csv_path}  ({len(df)} rows)")

    for ds_name in datasets:
        accuracy_vs_window(df, out_dir / f"acc_vs_window_{ds_name}.png", ds_name)
        if 2.0 in windows_s:
            itr_bar(df, out_dir / f"itr_bar_{ds_name}_2s.png", ds_name, 2.0)


if __name__ == "__main__":
    main()
