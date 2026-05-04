"""OpenBCI Cyton acquisition via BrainFlow, publishes to an LSL outlet."""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from .base import EEGSource, LSLPublisher

try:
    from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
except Exception:  # pragma: no cover
    BoardShim = None  # type: ignore
    BoardIds = None  # type: ignore
    BrainFlowInputParams = None  # type: ignore


class CytonSource(EEGSource, LSLPublisher):
    """8-channel Cyton at 250 Hz with USB dongle.

    Channels are taken from config (PO7..O2) but BrainFlow returns whatever the
    board sends; you map your electrode placement to those 8 inputs in
    docs/hardware_setup.md.
    """

    def __init__(self, serial_port: str, channels: Sequence[str],
                 fs: float = 250.0, stream_name: str = "ssvep_eeg"):
        if BoardShim is None:
            raise RuntimeError("brainflow is not installed")
        EEGSource.__init__(self, fs=fs, channels=channels, stream_name=stream_name)
        LSLPublisher.__init__(self, stream_name=stream_name, channels=channels, fs=fs)
        params = BrainFlowInputParams()
        params.serial_port = serial_port
        self._board = BoardShim(BoardIds.CYTON_BOARD.value, params)
        self._eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._board.prepare_session()
        self._board.start_stream()
        self._started = True
        time.sleep(0.5)

    def stop(self) -> None:
        if not self._started:
            return
        try:
            self._board.stop_stream()
        finally:
            self._board.release_session()
            self._started = False

    def read_chunk(self) -> np.ndarray:
        if not self._started:
            return np.empty((len(self.channels), 0), dtype=np.float32)
        n_avail = self._board.get_board_data_count()
        if n_avail <= 0:
            return np.empty((len(self.channels), 0), dtype=np.float32)
        data = self._board.get_board_data(n_avail)  # (rows, n_samples)
        eeg = data[self._eeg_channels[: len(self.channels)], :].astype(np.float32)
        self.push(eeg)
        return eeg
