"""Synthetic SSVEP source. Lets the pipeline run end-to-end without hardware."""
from __future__ import annotations

import threading
import time
from typing import Sequence

import numpy as np

from .base import EEGSource, LSLPublisher


class MockSource(EEGSource, LSLPublisher):
    """Generates band-limited noise plus a sinusoid at the active target.

    `set_target(idx)` switches which stim frequency dominates the synthetic
    signal — used by the live demo's mock harness to simulate a user looking
    at a different square.
    """

    def __init__(self, freqs: Sequence[float], channels: Sequence[str],
                 fs: float = 250.0, snr_db: float = 6.0, seed: int = 42,
                 stream_name: str = "ssvep_eeg"):
        EEGSource.__init__(self, fs=fs, channels=channels, stream_name=stream_name)
        LSLPublisher.__init__(self, stream_name=stream_name, channels=channels, fs=fs)
        self.freqs = np.asarray(freqs, dtype=float)
        self.snr_db = snr_db
        self._rng = np.random.default_rng(seed)
        # Phases must be stable across chunked synthesis calls so the producer
        # produces a phase-coherent sinusoid across thread iterations.
        self._channel_phases = self._rng.uniform(
            0, 2 * np.pi, size=(3, len(channels))
        )
        self._target_idx = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._buffer: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._start_time = 0.0
        self._n_pushed = 0

    def set_target(self, idx: int) -> None:
        """idx >= 0: SSVEP target index;  idx == -1: rest (pure noise)."""
        self._target_idx = int(idx) if idx == -1 else int(idx) % len(self.freqs)

    def set_schedule(self, schedule: list[tuple[float, int]]) -> None:
        """Time-sorted list of (t_seconds_since_start, target_idx).

        Used by speller demos to simulate the user looking at one cell, then a
        candidate, then another, etc. The producer thread picks up the scheduled
        target whenever its synth call falls past the threshold time.
        """
        self._schedule = sorted(schedule, key=lambda kv: kv[0])
        self._schedule_idx = 0

    def _maybe_advance_schedule(self, t_now: float) -> None:
        sch = getattr(self, "_schedule", None)
        if not sch:
            return
        i = getattr(self, "_schedule_idx", 0)
        while i < len(sch) and sch[i][0] <= t_now:
            self.set_target(sch[i][1])
            i += 1
        self._schedule_idx = i

    def synth(self, n_samples: int, t0: float) -> np.ndarray:
        """Synthesize n_samples. target_idx == -1 emits pure noise (rest)."""
        n_ch = len(self.channels)
        if self._target_idx < 0:
            # Rest / inter-trial: pure broadband noise, no SSVEP component.
            return self._rng.normal(0, 1.0, size=(n_ch, n_samples)).astype(np.float32)
        t = t0 + np.arange(n_samples) / self.fs
        f = self.freqs[self._target_idx]
        sig = np.zeros((n_ch, n_samples), dtype=np.float32)
        for hi, (h, amp) in enumerate([(1, 1.0), (2, 0.5), (3, 0.3)]):
            phase = self._channel_phases[hi, :n_ch][:, None]
            sig += amp * np.sin(2 * np.pi * h * f * t + phase).astype(np.float32)
        sig_power = np.mean(sig ** 2)
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = sig_power / max(snr_linear, 1e-9)
        noise = self._rng.normal(0, np.sqrt(noise_power), size=sig.shape).astype(np.float32)
        return sig + noise

    def _producer(self) -> None:
        chunk_samples = max(1, int(self.fs * 0.02))  # 20 ms chunks
        while self._running:
            now = time.time()
            t_elapsed = now - self._start_time
            self._maybe_advance_schedule(t_elapsed)
            target_n = int(t_elapsed * self.fs)
            n_new = target_n - self._n_pushed
            if n_new < chunk_samples:
                time.sleep(0.01)
                continue
            chunk = self.synth(n_new, t0=self._n_pushed / self.fs)
            self.push(chunk)
            with self._lock:
                self._buffer.append(chunk)
            self._n_pushed += n_new

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._start_time = time.time()
        self._n_pushed = 0
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def read_chunk(self) -> np.ndarray:
        with self._lock:
            if not self._buffer:
                return np.empty((len(self.channels), 0), dtype=np.float32)
            out = np.concatenate(self._buffer, axis=1)
            self._buffer.clear()
        return out
