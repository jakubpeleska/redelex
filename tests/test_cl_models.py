import torch
from torch_geometric.loader import NeighborLoader

from experiments.continuous_learning.models import HeterogeneousSAGE

from .helpers import NAN_TARGET_NUM_ROWS


def _model(hetero_graph):
    data, col_stats_dict = hetero_graph
    return data, HeterogeneousSAGE(data=data, col_stats_dict=col_stats_dict, gnn_channels=8)


def test_encoder_is_registered_once(hetero_graph):
    """`row_encoder` is an alias; a second registration stored the encoder twice."""
    _, model = _model(hetero_graph)
    keys = model.state_dict().keys()

    assert model.row_encoder is model.encoder
    assert not any(k.startswith("row_encoder.") for k in keys)
    assert any(k.startswith("encoder.") for k in keys)


def test_missing_numerical_cell_keeps_weights_finite(hetero_graph):
    """Regression: with the library NA default, one missing numerical cell made
    that column's weight NaN after the first Adam step."""
    data, model = _model(hetero_graph)
    # `users.target_num` holds NaN cells, so every batch over all users sees one.
    assert NAN_TARGET_NUM_ROWS
    loader = NeighborLoader(
        data, num_neighbors=[4, 2], input_nodes="users", batch_size=data["users"].num_nodes
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for _ in range(3):
        batch = next(iter(loader))
        out = model(batch, "users")
        loss = out.pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    assert bad == []
