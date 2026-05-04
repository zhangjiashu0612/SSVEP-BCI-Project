"""State-machine flow tests. The LM is loaded from bundled fallback so the
tests are deterministic without resources."""
from __future__ import annotations

from src.speller.layout import LETTERS, N_LETTERS
from src.speller.lm import LanguageModel
from src.speller.state import Layer, SpellerState


def _state() -> SpellerState:
    return SpellerState(lm=LanguageModel.from_resources(None))


def _letter_idx(ch: str) -> int:
    return LETTERS.index(ch)


def test_initial_state():
    s = _state()
    assert s.text_buffer == ""
    assert s.layer is Layer.LETTER
    assert s.candidates == []


def test_letter_w_populates_candidates_with_我():
    s = _state()
    t = s.on_target(_letter_idx("w"))
    assert t.layer is Layer.LETTER
    assert "我" in t.candidates


def test_pick_我_then_想要_then_喝水():
    s = _state()
    s.on_target(_letter_idx("w"))
    slot_我 = s.candidates.index("我")
    t1 = s.on_target(N_LETTERS + slot_我)
    assert t1.text_buffer == "我"
    assert t1.layer is Layer.CHAR
    assert "想要" in t1.candidates

    slot_想要 = s.candidates.index("想要")
    t2 = s.on_target(N_LETTERS + slot_想要)
    assert t2.text_buffer == "我想要"
    assert t2.layer is Layer.WORD
    assert "喝水" in t2.candidates

    slot_喝水 = s.candidates.index("喝水")
    t3 = s.on_target(N_LETTERS + slot_喝水)
    assert t3.text_buffer == "我想要喝水"


def test_letter_resets_to_LETTER_layer():
    s = _state()
    s.on_target(_letter_idx("w"))
    slot_我 = s.candidates.index("我")
    s.on_target(N_LETTERS + slot_我)  # text_buffer = "我", layer=CHAR
    assert s.layer is Layer.CHAR
    s.on_target(_letter_idx("h"))
    assert s.layer is Layer.LETTER  # new letter => back to LETTER layer
    assert s.text_buffer == "我"     # buffer unchanged


def test_empty_slot_ignored():
    s = _state()
    t = s.on_target(N_LETTERS + 0)  # candidate slot 0 with no candidates yet
    assert t.text_buffer == ""
    assert "ignore" in t.last_action


def test_out_of_range_target_ignored():
    s = _state()
    t = s.on_target(999)
    assert "ignore" in t.last_action
    assert t.text_buffer == ""


def test_callback_fires():
    seen = []
    s = _state()
    s.on_transition = lambda tr: seen.append(tr.last_action)
    s.on_target(_letter_idx("w"))
    assert seen and seen[-1].startswith("letter:")
