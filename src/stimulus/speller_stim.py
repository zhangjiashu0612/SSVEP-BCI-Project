"""32-target SSVEP speller UI.

Layout:
  +------------------------------------------------------+
  |  text output (current text_buffer, no flicker)        |
  +------------------------------------------------------+
  |  [cand0] [cand1] [cand2] [cand3] [cand4] [cand5]      |   <- 6 flickering
  |   闪烁词    闪烁词    ...                                |       candidate slots
  +------------------------------------------------------+
  |  a  b  c  d  e  f  g                                 |
  |  h  i  j  k  l  m  n                                 |   <- 26 flickering
  |  o  p  q  r  s  t  u                                 |       letter cells
  |  v  w  x  y  z                                       |
  +------------------------------------------------------+

Each cell is a `visual.Rect` whose fill is toggled per frame by a square wave
at its target frequency, with a `visual.TextStim` overlay for the label.
Highlight ring tracks the predicted target.
"""
from __future__ import annotations

import threading
from typing import Optional, Sequence

import numpy as np

from src.acquisition.base import lsl_marker_outlet


class SpellerStimulus:
    def __init__(self, freqs: Sequence[float], phases: Sequence[float],
                 letter_layout: list[list[str]], n_candidates: int = 6,
                 monitor_hz: int = 60, cell_size_px: int = 100,
                 candidate_size_px: tuple[int, int] = (200, 80),
                 fullscreen: bool = False, window_title: str = "SSVEP Speller"):
        from psychopy import visual, core, monitors  # lazy

        self.freqs = np.asarray(list(freqs), dtype=float)
        self.phases = np.asarray(list(phases), dtype=float)
        self.monitor_hz = monitor_hz
        self.n_candidates = n_candidates
        self.letter_layout = letter_layout
        self.cell_size_px = cell_size_px
        self.candidate_size_px = candidate_size_px

        n_targets = sum(len(r) for r in letter_layout) + n_candidates
        if len(self.freqs) != n_targets:
            raise ValueError(
                f"freqs length {len(self.freqs)} != expected {n_targets} "
                f"(letters + candidates)"
            )

        self.win = visual.Window(
            size=(1280, 800), fullscr=fullscreen, color=(-1, -1, -1),
            monitor=monitors.Monitor("default"), units="pix",
            allowGUI=True, waitBlanking=True, title=window_title,
        )

        cand_positions = self._candidate_positions()
        letter_positions = self._letter_positions()

        # Cells in TARGET INDEX order:
        #   first 26: letters (row-major from letter_layout)
        #   last 6: candidate slots
        all_positions = letter_positions + cand_positions
        all_labels = [c for row in letter_layout for c in row] \
            + ["" for _ in range(n_candidates)]
        self._cells: list = []
        self._labels: list = []
        for i, (pos, lab) in enumerate(zip(all_positions, all_labels)):
            is_cand = i >= len(letter_positions)
            w, h = (candidate_size_px if is_cand else (cell_size_px, cell_size_px))
            rect = visual.Rect(self.win, width=w, height=h,
                               fillColor=(1, 1, 1), pos=pos, lineWidth=0)
            text_height = 28 if is_cand else 36
            text = visual.TextStim(self.win, text=lab, pos=pos, color=(-1, -1, -1),
                                   height=text_height, bold=True)
            self._cells.append(rect)
            self._labels.append(text)

        self._highlight = visual.Rect(
            self.win, width=cell_size_px + 16, height=cell_size_px + 16,
            fillColor=None, lineColor=(1, 0.6, 0), lineWidth=4,
            pos=(0, 0),
        )
        self._highlight_idx: Optional[int] = None
        self._candidate_text: list[str] = ["" for _ in range(n_candidates)]
        self._buffer_text = ""
        self._lock = threading.Lock()

        self._buffer_stim = visual.TextStim(
            self.win, text="", pos=(0, 320), color=(1, 1, 1),
            height=40, bold=False, wrapWidth=1100,
        )
        self._marker = lsl_marker_outlet("ssvep_markers")
        self._clock = core.Clock()

    # ---- positioning -------------------------------------------------------

    def _letter_positions(self) -> list[tuple[float, float]]:
        size = self.cell_size_px
        gap = 24
        pitch = size + gap
        positions: list[tuple[float, float]] = []
        n_rows = len(self.letter_layout)
        # vertical center of grid below candidate row
        top_y = -40
        for r, row in enumerate(self.letter_layout):
            y = top_y - r * pitch
            row_w = (len(row) - 1) * pitch
            for c, _ in enumerate(row):
                x = -row_w / 2 + c * pitch
                positions.append((x, y))
        return positions

    def _candidate_positions(self) -> list[tuple[float, float]]:
        w, _ = self.candidate_size_px
        gap = 24
        pitch = w + gap
        total_w = (self.n_candidates - 1) * pitch
        y = 200
        return [(-total_w / 2 + k * pitch, y) for k in range(self.n_candidates)]

    # ---- public API --------------------------------------------------------

    def set_prediction(self, idx: Optional[int]) -> None:
        with self._lock:
            self._highlight_idx = idx

    def set_candidates(self, labels: list[str]) -> None:
        with self._lock:
            self._candidate_text = list(labels[: self.n_candidates])
            while len(self._candidate_text) < self.n_candidates:
                self._candidate_text.append("")

    def set_text_buffer(self, s: str) -> None:
        with self._lock:
            self._buffer_text = s

    def emit_marker(self, label: str) -> None:
        if self._marker is not None:
            self._marker.push_sample([label])

    def run(self, duration_s: float = 0.0) -> None:
        from psychopy.event import getKeys

        n_letters = sum(len(r) for r in self.letter_layout)
        self._clock.reset()
        self.emit_marker("speller_start")
        while True:
            t = self._clock.getTime()
            if duration_s and t >= duration_s:
                break

            # square-wave flicker per target
            phase_arg = 2 * np.pi * self.freqs * t + self.phases
            on = (np.cos(phase_arg) >= 0).astype(np.float32)
            colors = on * 2 - 1   # 1 or -1, shown as bright/dark grey

            with self._lock:
                buf = self._buffer_text
                cand_labels = list(self._candidate_text)
                hi = self._highlight_idx

            self._buffer_stim.text = buf
            self._buffer_stim.draw()

            # update candidate text labels
            for k in range(self.n_candidates):
                self._labels[n_letters + k].text = cand_labels[k]

            for i, rect in enumerate(self._cells):
                v = float(colors[i])
                rect.fillColor = (v, v, v)
                rect.draw()
                # only draw text overlay when cell is bright enough to read
                if v > 0:
                    self._labels[i].draw()

            if hi is not None and 0 <= hi < len(self._cells):
                self._highlight.size = (
                    self._cells[hi].width + 16,
                    self._cells[hi].height + 16,
                )
                self._highlight.pos = self._cells[hi].pos
                self._highlight.draw()

            self.win.flip()
            if "escape" in getKeys():
                break

        self.emit_marker("speller_stop")

    def close(self) -> None:
        try:
            self.win.close()
        except Exception:
            pass
