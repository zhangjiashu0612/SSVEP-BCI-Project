"""DSP filters used in both online and offline paths."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, cheby1, filtfilt, iirnotch, sosfiltfilt


def butter_bandpass_sos(low: float, high: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    return butter(order, [low / nyq, high / nyq], btype="band", output="sos")


def cheby1_bandpass_sos(low: float, high: float, fs: float, order: int = 6, rp: float = 0.5):
    nyq = 0.5 * fs
    high_eff = min(high, 0.99 * nyq)
    low_eff = max(low, 0.01)
    return cheby1(order, rp, [low_eff / nyq, high_eff / nyq], btype="band", output="sos")


def apply_sos(sos, x: np.ndarray, axis: int = -1) -> np.ndarray:
    return sosfiltfilt(sos, x, axis=axis)


def notch(x: np.ndarray, fs: float, freq: float = 60.0, q: float = 30.0, axis: int = -1) -> np.ndarray:
    b, a = iirnotch(freq, q, fs)
    return filtfilt(b, a, x, axis=axis)


def preprocess(x: np.ndarray, fs: float, low: float, high: float,
               order: int = 4, notch_hz: float | None = 60.0) -> np.ndarray:
    """Bandpass then optional notch. x: (channels, samples)."""
    sos = butter_bandpass_sos(low, high, fs, order=order)
    y = apply_sos(sos, x, axis=-1)
    if notch_hz:
        y = notch(y, fs, freq=notch_hz, axis=-1)
    return y
