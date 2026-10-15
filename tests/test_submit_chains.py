import pytest

from scripts.submit_chains import MAX_LANES, assign_lanes, chain_list


def test_chain_list_covers_the_grid():
    chains = chain_list(["rel-f1"], ["scratch_all", "cumulative"], [0, 1])
    assert len(chains) == 3 * 2 * 2
    assert ("rel-f1", "driver-top3", "cumulative", 1) in chains


def test_lanes_balance_longest_first_and_respect_the_cap():
    chains = [("d", "t", "m", s) for s in range(7)]
    hours = [10, 1, 1, 1, 1, 1, 1]
    lanes, loads = assign_lanes(chains, hours, 2)

    assert sorted(loads) == [6.0, 10.0]
    assert sum(len(lane) for lane in lanes) == 7
    with pytest.raises(ValueError, match=str(MAX_LANES)):
        assign_lanes(chains, hours, MAX_LANES + 1)
