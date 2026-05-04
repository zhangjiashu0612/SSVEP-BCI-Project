"""PsychoPy 4-square SSVEP stimulus.

Each square flickers by phase-locking to its target frequency f. We compute the
display contrast as cos(2*pi*f*t + phase) thresholded against 0 — for a 60 Hz
monitor and integer-divisor frequencies (7.5, 8.57, 10, 12 Hz) this is the
standard "joint frequency-and-phase modulation" stim. ON/OFF transitions and
the active prediction (set externally) are emitted on an LSL marker outlet.

Run standalone:  python -m src.stimulus.ssvep_stim
"""
from __future__ import annotations

import time
from threading import Lock
from typing import Optional, Sequence

from src.acquisition.base import lsl_marker_outlet
from src.utils.config import load_config


class SSVEPStimulus:
    def __init__(self, freqs: Sequence[float], phases: Sequence[float],
                 monitor_hz: int = 60, size_px: int = 200,
                 fullscreen: bool = False):
        from psychopy import visual, core, monitors  # imported lazily

        self.freqs = list(freqs)
        self.phases = list(phases)
        self.monitor_hz = monitor_hz
        self.win = visual.Window(
            size=(1200, 800), fullscr=fullscreen, color=(-1, -1, -1),
            monitor=monitors.Monitor("default"), units="pix",
            allowGUI=True, waitBlanking=True,
        )
        positions = [(-300, 200), (300, 200), (-300, -200), (300, -200)]
        self._squares = [
            visual.Rect(self.win, width=size_px, height=size_px,
                        fillColor=(1, 1, 1), pos=p, lineWidth=0)
            for p in positions[: len(self.freqs)]
        ]
        self._highlight = visual.Rect(
            self.win, width=size_px + 30, height=size_px + 30,
            fillColor=None, lineColor=(1, 0.5, 0), lineWidth=6,
            pos=positions[0],
        )
        self._highlight_idx: Optional[int] = None
        self._lock = Lock()
        self._marker = lsl_marker_outlet("ssvep_markers")
        self._clock = core.Clock()

    # public API consumed by live_demo
    def set_prediction(self, idx: Optional[int]) -> None:
        with self._lock:
            self._highlight_idx = idx

    def emit_marker(self, label: str) -> None:
        if self._marker is not None:
            self._marker.push_sample([label])

    def run(self, duration_s: float = 0.0) -> None:
        from psychopy.event import getKeys

        self._clock.reset()
        self.emit_marker("stim_start")
        while True:
            t = self._clock.getTime()
            if duration_s and t >= duration_s:
                break
            for i, sq in enumerate(self._squares):
                phase = self.phases[i] if i < len(self.phases) else 0.0
                val = (1 + (1 if (((2 * 3.14159265 * self.freqs[i] * t + phase)
                                   % (2 * 3.14159265)) < 3.14159265) else -1)) / 1
                sq.fillColor = (val, val, val)
                sq.draw()
            with self._lock:
                hi = self._highlight_idx
            if hi is not None and 0 <= hi < len(self._squares):
                self._highlight.pos = self._squares[hi].pos
                self._highlight.draw()
            self.win.flip()
            if "escape" in getKeys():
                break
        self.emit_marker("stim_stop")

    def close(self) -> None:
        self.win.close()


def main() -> None:
    cfg = load_config()
    s = SSVEPStimulus(
        freqs=cfg["stimulus"]["freqs_hz"],
        phases=cfg["stimulus"]["phases_rad"],
        monitor_hz=cfg["stimulus"]["monitor_hz"],
        size_px=cfg["stimulus"]["size_px"],
        fullscreen=cfg["stimulus"]["fullscreen"],
    )
    try:
        s.run(duration_s=cfg["stimulus"]["duration_s"])
    finally:
        s.close()


if __name__ == "__main__":
    main()
