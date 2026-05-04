"""32-target frequency table for the 26-letter + 6-candidate speller.

We use 32 distinct frequencies at 0.2 Hz spacing in the 8.0–14.2 Hz band —
the spacing is well-resolvable by FBCCA at a 2.5 s window and reproduces the
density Wang2016 used in the original 40-target SSVEP benchmark. Phase coding
is intentionally NOT used because standard sin+cos CCA reference signals span
a phase-invariant 2D subspace at each harmonic; phase discrimination requires
a learned decoder (TRCA on per-subject templates), which contradicts the
"clone the repo and run mock instantly" portfolio constraint.

Phases are still threaded through the API as zeros so a future TRCA-based
JFPM extension can drop in without changing call sites.

The mapping idx → (letter | candidate_slot) is fixed so the rest of the
speller (state machine, UI, config) can refer to targets by integer index:

    idx  0..25  → letters a..z
    idx 26..31  → candidate slots 0..5
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

LETTERS = tuple("abcdefghijklmnopqrstuvwxyz")
N_LETTERS = len(LETTERS)
N_CANDIDATES_DEFAULT = 6
N_TARGETS_DEFAULT = N_LETTERS + N_CANDIDATES_DEFAULT  # 32


@dataclass(frozen=True)
class TargetTable:
    """Frozen view of the speller's flicker targets."""
    freqs: np.ndarray   # (n_targets,)
    phases: np.ndarray  # (n_targets,) radians
    labels: tuple[str, ...]  # human-readable: 'a'..'z', 'cand0'..'cand5'

    def __len__(self) -> int:
        return len(self.freqs)

    @property
    def n_letters(self) -> int:
        return sum(1 for lab in self.labels if len(lab) == 1)

    @property
    def n_candidates(self) -> int:
        return len(self.labels) - self.n_letters

    def is_letter(self, idx: int) -> bool:
        return 0 <= idx < self.n_letters

    def is_candidate(self, idx: int) -> bool:
        return self.n_letters <= idx < len(self.labels)

    def candidate_slot(self, idx: int) -> int:
        if not self.is_candidate(idx):
            raise ValueError(f"idx {idx} is not a candidate slot")
        return idx - self.n_letters

    def letter(self, idx: int) -> str:
        if not self.is_letter(idx):
            raise ValueError(f"idx {idx} is not a letter")
        return self.labels[idx]


def freq_grid(low_hz: float = 8.0, step_hz: float = 0.2,
              n_targets: int = N_TARGETS_DEFAULT,
              n_candidates: int = N_CANDIDATES_DEFAULT) -> TargetTable:
    """32 distinct frequencies at fixed spacing — Wang2016-style frequency grid.

    Defaults: 8.0, 8.2, ..., 14.2 Hz (32 values). All phases zero — see module
    docstring on why JFPM phase-coding is not used here.
    """
    if n_targets < N_LETTERS + n_candidates:
        raise ValueError(
            f"n_targets={n_targets} can't host {N_LETTERS} letters + "
            f"{n_candidates} candidates"
        )
    freqs = low_hz + step_hz * np.arange(n_targets)
    phases = np.zeros(n_targets)
    labels = list(LETTERS) + [f"cand{k}" for k in range(n_candidates)]
    return TargetTable(freqs=freqs, phases=phases, labels=tuple(labels))


# Backwards-compat alias kept for any code that imports the old name.
def jfpm_table(*args, **kwargs) -> TargetTable:  # pragma: no cover
    return freq_grid(*args, **kwargs)
