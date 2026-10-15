import numpy as np
import pytest

from redelex.continual.replay import ReservoirBuffer


def test_fills_then_stays_at_capacity():
    buf = ReservoirBuffer(capacity=5, seed=0)
    buf.add([1, 2, 3])
    assert buf.row_ids.tolist() == [1, 2, 3] and buf.seen == 3

    buf.add(range(10, 100))
    assert len(buf) == 5 and buf.seen == 93
    assert set(buf.row_ids) <= {1, 2, 3} | set(range(10, 100))


def test_every_row_is_kept_with_equal_probability():
    hits = np.zeros(100)
    for seed in range(2000):
        buf = ReservoirBuffer(capacity=10, seed=seed)
        buf.add(range(50))  # an early episode
        buf.add(range(50, 100))  # a later one
        hits[buf.row_ids] += 1
    rate = hits / 2000
    assert rate == pytest.approx(np.full(100, 0.1), abs=0.03)


def test_state_round_trip_continues_identically():
    a = ReservoirBuffer(capacity=8, seed=3)
    a.add(range(40))
    b = ReservoirBuffer.from_state_dict(a.state_dict())

    a.add(range(40, 80))
    b.add(range(40, 80))
    assert a.row_ids.tolist() == b.row_ids.tolist() and a.seen == b.seen


def test_rejects_non_positive_capacity():
    with pytest.raises(ValueError):
        ReservoirBuffer(0)
