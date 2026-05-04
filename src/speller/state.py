"""Speller state machine.

Consumes confirmed target indices from the SSVEP pipeline and produces UI
updates: text buffer changes plus the new candidate-row labels. The machine
is a thin wrapper around the LM — all logic lives here, no NLP in the UI.

Layers:
    layer 0 (LETTER):  awaiting a letter; prefix = letters typed so far
    layer 1 (CHAR):    a single Chinese char was committed; candidates = words
    layer 2 (WORD):    a word was committed; candidates = bigram continuations

Target index conventions (must match TargetTable from layout.py):
    0..25  → letters a..z  → triggers a letter event
    26..31 → candidate slots 0..5 → triggers a candidate-pick event

Returned `Transition` describes everything the UI needs to redraw.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .layout import LETTERS, N_LETTERS
from .lm import LanguageModel


class Layer(str, Enum):
    LETTER = "letter"
    CHAR = "char"
    WORD = "word"


@dataclass
class Transition:
    text_buffer: str
    candidates: list[str]
    layer: Layer
    last_action: str  # human-readable: "letter:w", "candidate:0->我", "ignore", ...


@dataclass
class SpellerState:
    lm: LanguageModel
    n_candidate_slots: int = 6
    text_buffer: str = ""
    candidates: list[str] = field(default_factory=list)
    layer: Layer = Layer.LETTER
    letter_prefix: str = ""  # chars typed since last commit, e.g. "wo"
    history: list[str] = field(default_factory=list)
    on_transition: Callable[[Transition], None] | None = None

    def reset(self) -> None:
        self.text_buffer = ""
        self.candidates = []
        self.layer = Layer.LETTER
        self.letter_prefix = ""
        self.history.clear()

    # ---- entry point -------------------------------------------------------

    def on_target(self, idx: int) -> Transition:
        if 0 <= idx < N_LETTERS:
            t = self._on_letter(LETTERS[idx])
        elif N_LETTERS <= idx < N_LETTERS + self.n_candidate_slots:
            t = self._on_candidate(idx - N_LETTERS)
        else:
            t = self._snapshot("ignore:out-of-range")
        if self.on_transition is not None:
            self.on_transition(t)
        return t

    # ---- letter event ------------------------------------------------------

    def _on_letter(self, letter: str) -> Transition:
        self.letter_prefix = self.letter_prefix + letter
        chars = self.lm.predict_char(self.letter_prefix, self.n_candidate_slots)
        # If the new prefix doesn't have any matches but a single-letter prefix
        # would, treat the letter as starting a fresh prefix instead. This
        # makes the user's life easier when LM coverage is patchy.
        if not chars:
            chars = self.lm.predict_char(letter, self.n_candidate_slots)
            self.letter_prefix = letter
        self.candidates = _pad(chars, self.n_candidate_slots)
        self.layer = Layer.LETTER
        self.history.append(f"letter:{self.letter_prefix}")
        return self._snapshot(f"letter:{self.letter_prefix}")

    # ---- candidate event ---------------------------------------------------

    def _on_candidate(self, slot: int) -> Transition:
        if slot < 0 or slot >= self.n_candidate_slots:
            return self._snapshot("ignore:bad-slot")
        if slot >= len(self.candidates) or not self.candidates[slot]:
            return self._snapshot("ignore:empty-slot")

        chosen = self.candidates[slot]
        self.text_buffer += chosen
        self.letter_prefix = ""
        self.history.append(f"candidate:{slot}->{chosen}")

        if self.layer is Layer.LETTER:
            # Picked a single character — promote to CHAR layer, propose words
            words = self.lm.predict_word(chosen, self.n_candidate_slots)
            self.layer = Layer.CHAR
            self.candidates = _pad(words, self.n_candidate_slots)
        elif self.layer is Layer.CHAR:
            # Picked a word — promote to WORD layer, propose continuations
            cont = self.lm.predict_continuation(chosen, self.n_candidate_slots)
            self.layer = Layer.WORD
            self.candidates = _pad(cont, self.n_candidate_slots)
        else:  # Layer.WORD — keep proposing continuations of the new word
            cont = self.lm.predict_continuation(chosen, self.n_candidate_slots)
            self.candidates = _pad(cont, self.n_candidate_slots)
        return self._snapshot(f"candidate:{slot}->{chosen}")

    # ---- helpers -----------------------------------------------------------

    def _snapshot(self, action: str) -> Transition:
        return Transition(
            text_buffer=self.text_buffer,
            candidates=list(self.candidates),
            layer=self.layer,
            last_action=action,
        )


def _pad(items: list[str], n: int) -> list[str]:
    items = list(items[:n])
    while len(items) < n:
        items.append("")
    return items
