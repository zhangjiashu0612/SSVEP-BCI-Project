"""Real-time SSVEP pipeline.

Producer thread keeps a ring buffer fed; consumer thread pulls the most recent
`window_s` worth of samples every `step_ms`, runs preprocessing + classifier,
and applies majority voting before publishing a "confirmed" prediction.

The producer is decoupled from the data source: callers pass a chunk_fn
(returning (n_channels, n_new) arrays) and an optional LSL inlet path is
provided as a convenience for the live demo.
"""
from __future__ import annotations

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from src.algos.base import Classifier
from src.processing.filters import preprocess


class RingBuffer:
    def __init__(self, n_channels: int, capacity_samples: int):
        self.n_channels = n_channels
        self.capacity = capacity_samples
        self.buf = np.zeros((n_channels, capacity_samples), dtype=np.float32)
        self.write_idx = 0
        self.n_written = 0
        self.lock = threading.Lock()

    def push(self, chunk: np.ndarray) -> None:
        if chunk.size == 0:
            return
        nch, n = chunk.shape
        if nch != self.n_channels:
            raise ValueError(f"channel mismatch: got {nch} expected {self.n_channels}")
        with self.lock:
            end = self.write_idx + n
            if end <= self.capacity:
                self.buf[:, self.write_idx:end] = chunk
            else:
                first = self.capacity - self.write_idx
                self.buf[:, self.write_idx:] = chunk[:, :first]
                self.buf[:, : n - first] = chunk[:, first:]
            self.write_idx = end % self.capacity
            self.n_written += n

    def latest(self, n_samples: int) -> Optional[np.ndarray]:
        if n_samples > self.capacity:
            return None
        with self.lock:
            if self.n_written < n_samples:
                return None
            start = (self.write_idx - n_samples) % self.capacity
            if start + n_samples <= self.capacity:
                return self.buf[:, start:start + n_samples].copy()
            first = self.capacity - start
            return np.concatenate(
                [self.buf[:, start:], self.buf[:, : n_samples - first]], axis=1
            )


@dataclass
class Prediction:
    raw_idx: int
    confirmed_idx: Optional[int]
    score_freq_hz: float
    latency_ms: float
    timestamp: float


class SSVEPPipeline:
    def __init__(self, classifier: Classifier, fs: float, n_channels: int,
                 window_s: float = 2.0, step_ms: int = 200,
                 ring_buffer_s: float = 6.0, vote_window: int = 3,
                 bandpass=(6.0, 60.0), notch_hz: float | None = 60.0,
                 filter_order: int = 4):
        self.classifier = classifier
        self.fs = fs
        self.n_channels = n_channels
        self.window_samples = int(round(window_s * fs))
        self.step_s = step_ms / 1000.0
        self.vote_window = vote_window
        self.bandpass = bandpass
        self.notch_hz = notch_hz
        self.filter_order = filter_order
        self.buffer = RingBuffer(n_channels, int(round(ring_buffer_s * fs)))
        self._running = False
        self._producer_thread: Optional[threading.Thread] = None
        self._consumer_thread: Optional[threading.Thread] = None
        self._votes: deque = deque(maxlen=vote_window)
        self._predictions: list[Prediction] = []
        self._lock = threading.Lock()
        self._on_prediction: Optional[Callable[[Prediction], None]] = None

    def on_prediction(self, fn: Callable[[Prediction], None]) -> None:
        self._on_prediction = fn

    @property
    def predictions(self) -> list[Prediction]:
        with self._lock:
            return list(self._predictions)

    # -- producer ---------------------------------------------------------
    def _producer_loop(self, chunk_fn: Callable[[], np.ndarray]) -> None:
        while self._running:
            chunk = chunk_fn()
            if chunk is not None and chunk.size > 0:
                self.buffer.push(chunk)
            time.sleep(0.005)

    # -- consumer ---------------------------------------------------------
    def _consume_once(self) -> Optional[Prediction]:
        win = self.buffer.latest(self.window_samples)
        if win is None:
            return None
        t0 = time.perf_counter()
        x = preprocess(win, self.fs, self.bandpass[0], self.bandpass[1],
                       order=self.filter_order, notch_hz=self.notch_hz)
        x = x - x.mean(axis=1, keepdims=True)
        idx = int(self.classifier.predict(x[None])[0])
        latency_ms = (time.perf_counter() - t0) * 1000
        self._votes.append(idx)
        confirmed = None
        if len(self._votes) == self.vote_window:
            count = Counter(self._votes)
            top, n = count.most_common(1)[0]
            if n == self.vote_window:
                confirmed = int(top)
        f0 = float(self.classifier.freqs[idx])
        pred = Prediction(raw_idx=idx, confirmed_idx=confirmed, score_freq_hz=f0,
                          latency_ms=latency_ms, timestamp=time.time())
        with self._lock:
            self._predictions.append(pred)
        if self._on_prediction is not None:
            try:
                self._on_prediction(pred)
            except Exception:
                pass
        return pred

    def _consumer_loop(self) -> None:
        while self._running:
            t0 = time.time()
            self._consume_once()
            slept = time.time() - t0
            if slept < self.step_s:
                time.sleep(self.step_s - slept)

    # -- lifecycle --------------------------------------------------------
    def start(self, chunk_fn: Callable[[], np.ndarray]) -> None:
        if self._running:
            return
        self._running = True
        self._producer_thread = threading.Thread(
            target=self._producer_loop, args=(chunk_fn,), daemon=True
        )
        self._consumer_thread = threading.Thread(
            target=self._consumer_loop, daemon=True
        )
        self._producer_thread.start()
        self._consumer_thread.start()

    def stop(self) -> None:
        self._running = False
        for t in (self._producer_thread, self._consumer_thread):
            if t is not None:
                t.join(timeout=1.5)
        self._producer_thread = None
        self._consumer_thread = None


def lsl_chunk_fn(stream_name: str = "ssvep_eeg", n_channels: int = 8):
    """Build a chunk_fn that pulls from an LSL inlet.

    Returns (chunk_fn, close_fn). Used by live_demo when the source publishes
    over LSL — keeps the producer one process away from the acquisition.
    """
    from pylsl import StreamInlet, resolve_byprop

    streams = resolve_byprop("name", stream_name, timeout=5.0)
    if not streams:
        raise RuntimeError(f"LSL stream '{stream_name}' not found")
    inlet = StreamInlet(streams[0], max_chunklen=64)

    def fn() -> np.ndarray:
        samples, _ = inlet.pull_chunk(timeout=0.0, max_samples=512)
        if not samples:
            return np.empty((n_channels, 0), dtype=np.float32)
        arr = np.asarray(samples, dtype=np.float32).T
        return arr

    return fn, lambda: inlet.close_stream()
