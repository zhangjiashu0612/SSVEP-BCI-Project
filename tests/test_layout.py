import numpy as np
import pytest

from src.speller.layout import LETTERS, freq_grid


def test_freq_grid_default_size():
    tab = freq_grid()
    assert len(tab) == 32
    assert tab.n_letters == 26
    assert tab.n_candidates == 6


def test_freq_grid_distinct_frequencies():
    tab = freq_grid()
    assert len(set(tab.freqs.tolist())) == 32, "every frequency must be distinct"


def test_freq_grid_phases_all_zero():
    tab = freq_grid()
    assert np.allclose(tab.phases, 0.0)


def test_freq_grid_label_routing():
    tab = freq_grid()
    for i, lab in enumerate(LETTERS):
        assert tab.is_letter(i)
        assert tab.letter(i) == lab
    for s in range(6):
        assert tab.is_candidate(26 + s)
        assert tab.candidate_slot(26 + s) == s


def test_freq_grid_too_few_targets():
    with pytest.raises(ValueError):
        freq_grid(n_targets=10)


def test_freq_grid_spacing():
    tab = freq_grid(low_hz=8.0, step_hz=0.2)
    assert tab.freqs[0] == pytest.approx(8.0)
    assert tab.freqs[-1] == pytest.approx(8.0 + 0.2 * 31)
    diffs = np.diff(tab.freqs)
    assert np.allclose(diffs, 0.2)
