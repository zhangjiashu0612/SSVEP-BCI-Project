"""LM smoke tests — work entirely from the bundled fallback dicts so they
pass on a fresh clone with no resources built."""
from __future__ import annotations

from src.speller.lm import LanguageModel


def test_lm_loads_with_no_resources():
    lm = LanguageModel.from_resources(None)
    assert lm.char_freq, "fallback char_freq must be populated"


def test_predict_char_for_w_includes_我():
    lm = LanguageModel.from_resources(None)
    chars = lm.predict_char("w", 6)
    assert "我" in chars, f"expected 我 among predict_char('w'), got {chars}"
    assert len(chars) == 6


def test_predict_char_empty_prefix():
    lm = LanguageModel.from_resources(None)
    assert lm.predict_char("", 6) == []
    assert lm.predict_char("123", 6) == []


def test_predict_word_for_我_includes_想要():
    lm = LanguageModel.from_resources(None)
    words = lm.predict_word("我", 6)
    assert "想要" in words


def test_predict_continuation_for_想要_includes_喝水():
    lm = LanguageModel.from_resources(None)
    cont = lm.predict_continuation("想要", 6)
    assert "喝水" in cont


def test_predict_continuation_unknown_word_is_empty():
    lm = LanguageModel.from_resources(None)
    assert lm.predict_continuation("不存在的词", 6) == []


def test_predict_char_for_n_returns_chars():
    lm = LanguageModel.from_resources(None)
    chars = lm.predict_char("n", 6)
    # pypinyin dict should give *something* for "n" prefix even if our seed
    # didn't include any 'n'-prefix chars.
    assert len(chars) > 0
