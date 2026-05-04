import math

from src.utils.metrics import wolpaw_itr


def test_wolpaw_itr_perfect():
    assert math.isclose(wolpaw_itr(1.0, 4, 2.0), math.log2(4) * 60 / 2, rel_tol=1e-9)


def test_wolpaw_itr_chance_or_below():
    assert wolpaw_itr(0.25, 4, 2.0) == 0.0
    assert wolpaw_itr(0.10, 4, 2.0) == 0.0


def test_wolpaw_itr_increases_with_accuracy():
    a = wolpaw_itr(0.6, 4, 2.0)
    b = wolpaw_itr(0.8, 4, 2.0)
    c = wolpaw_itr(0.95, 4, 2.0)
    assert a < b < c
