import numpy as np
import pytest
import torch

from redelex.loaders import MixtureLoader, Source, epoch_steps, mixture_weights
from redelex.transforms import AttachTargetTransform


def _source(n, offset, share, n_users):
    nodes = (torch.arange(n) + offset) % n_users
    # Anchors late enough that every user exists; targets encode the source.
    time = torch.full((n,), 1_650_000_000, dtype=torch.long)
    target = torch.full((n,), float(offset))
    return Source(nodes=nodes, time=time, target=target, share=share)


def _loader(stream_graph, sources, num_samples=8_000, seed=0):
    data, _ = stream_graph
    return MixtureLoader(
        data,
        "users",
        sources,
        num_neighbors=[4, 2],
        batch_size=128,
        num_samples=num_samples,
        seed=seed,
        transform_factory=AttachTargetTransform,
    )


def test_weights_give_exact_expected_shares():
    w = mixture_weights({"new": 6, "old": 994}, {"new": 0.5, "old": 0.5})

    assert w[:6].sum() == pytest.approx(0.5)
    assert w[6:].sum() == pytest.approx(0.5)
    # An empty source hands its share to the rest.
    w = mixture_weights({"new": 0, "old": 10}, {"new": 0.5, "old": 0.5})
    assert w.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        mixture_weights({"new": 0}, {"new": 1.0})


@pytest.mark.parametrize("p", [0.25, 0.5, 0.75, 1.0])
def test_sample_share_matches_p_even_for_a_tiny_new_source(stream_graph, p):
    # 50 new rows: under batch-level mixing their partial batch would skew the share.
    n_users = stream_graph[0]["users"].num_nodes
    loader = _loader(
        stream_graph,
        {"new": _source(50, 0, p, n_users), "old": _source(2000, 1000, 1 - p, n_users)},
    )

    counts = {"new": 0, "old": 0}
    for batch in loader:
        for k, v in loader.source_counts(batch).items():
            counts[k] += v
        # Targets travel with their rows.
        y = batch["users"].y
        src = loader.source_of_row[batch["users"].input_id.numpy()]
        assert torch.equal(y, torch.where(torch.as_tensor(src == 0), 0.0, 1000.0))

    total = counts["new"] + counts["old"]
    assert total == 8_000
    assert counts["new"] / total == pytest.approx(p, abs=0.02)


def test_seed_fixes_the_draw(stream_graph):
    n_users = stream_graph[0]["users"].num_nodes
    srcs = lambda: {"all": _source(500, 0, 1.0, n_users)}  # noqa: E731
    a = [b["users"].input_id for b in _loader(stream_graph, srcs(), 1000, seed=3)]
    b = [b["users"].input_id for b in _loader(stream_graph, srcs(), 1000, seed=3)]
    c = [b["users"].input_id for b in _loader(stream_graph, srcs(), 1000, seed=4)]

    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not all(torch.equal(x, y) for x, y in zip(a, c))


def test_epoch_steps():
    assert epoch_steps(690, 128) == 6
    assert epoch_steps(690, 128, fraction=0.25) == 2
    assert epoch_steps(10, 128, fraction=0.25, minimum=5) == 5
    assert np.isfinite(epoch_steps(10**7, 128))
