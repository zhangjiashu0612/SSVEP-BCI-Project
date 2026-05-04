"""Common interface for EEG sources that publish to an LSL outlet."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

try:  # pylsl is optional for unit tests that don't need streaming
    from pylsl import StreamInfo, StreamOutlet, local_clock
except Exception:  # pragma: no cover
    StreamInfo = StreamOutlet = None  # type: ignore
    local_clock = time.time  # type: ignore


class EEGSource(ABC):
    """Source produces (n_channels, n_new_samples) chunks at `fs`.

    The mixin `LSLPublisher` adds an outlet so the same code path is used by
    cyton, mock, and replay without each source caring about LSL.
    """

    def __init__(self, fs: float, channels: Sequence[str], stream_name: str):
        self.fs = float(fs)
        self.channels = list(channels)
        self.stream_name = stream_name

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def read_chunk(self) -> np.ndarray:
        """Return new samples shaped (n_channels, n_new_samples)."""

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


class LSLPublisher:
    """Mixin: holds an LSL outlet and pushes chunks."""

    def __init__(self, *, stream_name: str, channels: Sequence[str], fs: float,
                 stream_type: str = "EEG"):
        if StreamOutlet is None:
            self._outlet = None
            return
        info = StreamInfo(stream_name, stream_type, len(channels), fs, "float32",
                          f"{stream_name}-uid")
        chans = info.desc().append_child("channels")
        for ch in channels:
            c = chans.append_child("channel")
            c.append_child_value("label", ch)
            c.append_child_value("unit", "microvolts")
            c.append_child_value("type", "EEG")
        self._outlet = StreamOutlet(info, chunk_size=32, max_buffered=360)

    def push(self, chunk: np.ndarray) -> None:
        """chunk: (n_channels, n_samples)."""
        if self._outlet is None or chunk.size == 0:
            return
        self._outlet.push_chunk(chunk.T.astype(np.float32).tolist())


def lsl_marker_outlet(stream_name: str = "ssvep_markers") -> "StreamOutlet | None":
    if StreamOutlet is None:
        return None
    info = StreamInfo(stream_name, "Markers", 1, 0, "string", f"{stream_name}-uid")
    return StreamOutlet(info)
