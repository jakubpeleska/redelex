import pytest
import torch
from torch import nn

from redelex.continual.regularization import ParameterAnchor, fisher_diagonal


@pytest.fixture
def model():
    torch.manual_seed(0)
    return nn.Linear(3, 1)


def _batches(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(8, 3, generator=g), torch.randn(8, 1, generator=g)) for _ in range(n)]


FORWARD = lambda m, b: m(b[0])
TARGET = lambda b: b[1]
LOSS = nn.MSELoss()


# --- fisher_diagonal --------------------------------------------------------


def test_fisher_covers_every_trainable_parameter(model):
    fisher = fisher_diagonal(model, _batches(), LOSS, FORWARD, TARGET)
    assert set(fisher) == {name for name, _ in model.named_parameters()}
    for name, value in fisher.items():
        assert value.shape == dict(model.named_parameters())[name].shape


def test_fisher_is_non_negative(model):
    fisher = fisher_diagonal(model, _batches(), LOSS, FORWARD, TARGET)
    for value in fisher.values():
        assert torch.all(value >= 0)


def test_fisher_is_nonzero_for_a_learning_signal(model):
    fisher = fisher_diagonal(model, _batches(), LOSS, FORWARD, TARGET)
    assert sum(v.sum() for v in fisher.values()) > 0


def test_fisher_respects_max_batches(model):
    many = fisher_diagonal(model, _batches(n=8), LOSS, FORWARD, TARGET)
    few = fisher_diagonal(model, _batches(n=8), LOSS, FORWARD, TARGET, max_batches=2)
    # different amounts of data give different estimates
    assert not torch.allclose(many["weight"], few["weight"])


def test_fisher_zero_when_no_batches(model):
    fisher = fisher_diagonal(model, [], LOSS, FORWARD, TARGET)
    for value in fisher.values():
        assert torch.all(value == 0)


def test_fisher_leaves_gradients_clean(model):
    fisher_diagonal(model, _batches(), LOSS, FORWARD, TARGET)
    for param in model.parameters():
        assert param.grad is None


def test_fisher_restores_training_mode(model):
    model.train()
    fisher_diagonal(model, _batches(), LOSS, FORWARD, TARGET)
    assert model.training
    model.eval()
    fisher_diagonal(model, _batches(), LOSS, FORWARD, TARGET)
    assert not model.training


def test_fisher_ignores_frozen_parameters(model):
    model.bias.requires_grad_(False)
    fisher = fisher_diagonal(model, _batches(), LOSS, FORWARD, TARGET)
    assert "bias" not in fisher and "weight" in fisher


# --- ParameterAnchor --------------------------------------------------------


def test_penalty_is_zero_before_any_consolidation(model):
    assert ParameterAnchor(lam=1.0).penalty(model).item() == 0.0


def test_penalty_is_zero_at_the_anchor_point(model):
    anchor = ParameterAnchor(lam=1.0)
    fisher = fisher_diagonal(model, _batches(), LOSS, FORWARD, TARGET)
    anchor.consolidate(model, fisher)
    # parameters have not moved since consolidation
    assert anchor.penalty(model).item() == pytest.approx(0.0, abs=1e-9)


def test_penalty_grows_as_parameters_drift(model):
    anchor = ParameterAnchor(lam=1.0)
    anchor.consolidate(model, fisher_diagonal(model, _batches(), LOSS, FORWARD, TARGET))

    with torch.no_grad():
        model.weight += 0.1
    near = anchor.penalty(model).item()
    with torch.no_grad():
        model.weight += 0.9
    far = anchor.penalty(model).item()
    assert 0 < near < far


def test_penalty_scales_with_lambda(model):
    fisher = fisher_diagonal(model, _batches(), LOSS, FORWARD, TARGET)
    weak, strong = ParameterAnchor(lam=1.0), ParameterAnchor(lam=10.0)
    weak.consolidate(model, fisher)
    strong.consolidate(model, fisher)
    with torch.no_grad():
        model.weight += 0.5
    assert strong.penalty(model).item() == pytest.approx(10 * weak.penalty(model).item())


def test_lambda_zero_disables_the_term(model):
    anchor = ParameterAnchor(lam=0.0)
    anchor.consolidate(model, fisher_diagonal(model, _batches(), LOSS, FORWARD, TARGET))
    with torch.no_grad():
        model.weight += 1.0
    assert anchor.penalty(model).item() == 0.0


def test_penalty_weights_important_parameters_more(model):
    # a parameter with 100x the Fisher should dominate an equal displacement
    anchor = ParameterAnchor(lam=1.0)
    fisher = {"weight": torch.tensor([[100.0, 1.0, 1.0]]), "bias": torch.zeros(1)}
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    anchor.consolidate(model, fisher)

    with torch.no_grad():
        model.weight[0, 0] = 1.0
    important = anchor.penalty(model).item()
    with torch.no_grad():
        model.weight.zero_()
        model.weight[0, 1] = 1.0
    unimportant = anchor.penalty(model).item()
    assert important == pytest.approx(100 * unimportant)


def test_penalty_is_differentiable(model):
    anchor = ParameterAnchor(lam=1.0)
    anchor.consolidate(model, fisher_diagonal(model, _batches(), LOSS, FORWARD, TARGET))
    with torch.no_grad():
        model.weight += 0.3
    anchor.penalty(model).backward()
    assert model.weight.grad is not None and torch.any(model.weight.grad != 0)


def test_consolidate_accumulates_across_episodes(model):
    anchor = ParameterAnchor(lam=1.0, gamma=1.0)
    fisher = {"weight": torch.ones_like(model.weight), "bias": torch.ones_like(model.bias)}
    anchor.consolidate(model, fisher)
    first = anchor.fisher["weight"].clone()
    anchor.consolidate(model, fisher)
    torch.testing.assert_close(anchor.fisher["weight"], 2 * first)
    assert anchor.episodes == 2


def test_gamma_decays_older_importances(model):
    anchor = ParameterAnchor(lam=1.0, gamma=0.5)
    fisher = {"weight": torch.ones_like(model.weight), "bias": torch.ones_like(model.bias)}
    anchor.consolidate(model, fisher)
    anchor.consolidate(model, fisher)
    # 0.5 * 1 + 1 = 1.5, rather than 2
    torch.testing.assert_close(anchor.fisher["weight"], 1.5 * torch.ones_like(model.weight))


def test_consolidate_reanchors_to_current_weights(model):
    anchor = ParameterAnchor(lam=1.0)
    fisher = {"weight": torch.ones_like(model.weight), "bias": torch.ones_like(model.bias)}
    anchor.consolidate(model, fisher)
    with torch.no_grad():
        model.weight += 1.0
    anchor.consolidate(model, fisher)
    # the anchor moved with the model, so the penalty is zero again
    assert anchor.penalty(model).item() == pytest.approx(0.0, abs=1e-9)


def test_anchor_round_trips_through_state_dict(model):
    anchor = ParameterAnchor(lam=2.5, gamma=0.9)
    anchor.consolidate(model, fisher_diagonal(model, _batches(), LOSS, FORWARD, TARGET))
    with torch.no_grad():
        model.weight += 0.4

    restored = ParameterAnchor.from_state_dict(anchor.state_dict())
    assert restored.lam == 2.5 and restored.gamma == 0.9 and restored.episodes == 1
    assert restored.penalty(model).item() == pytest.approx(anchor.penalty(model).item())


def test_anchor_skips_parameters_whose_shape_changed(model):
    anchor = ParameterAnchor(lam=1.0)
    anchor.consolidate(model, {"weight": torch.ones(1, 3), "bias": torch.ones(1)})
    wider = nn.Linear(5, 1)  # architecture changed under the anchor
    # the mismatched parameter is skipped rather than raising
    assert anchor.penalty(wider).item() >= 0.0


@pytest.mark.parametrize("lam,gamma", [(-1.0, 1.0), (1.0, -0.1), (1.0, 1.5)])
def test_anchor_rejects_bad_hyperparameters(lam, gamma):
    with pytest.raises(ValueError):
        ParameterAnchor(lam=lam, gamma=gamma)


# --- regression: device agreement across an episode boundary -----------------


def test_consolidate_accepts_a_fisher_on_another_device(model):
    # An anchor restored from disk holds CPU tensors (state_dict moves them there),
    # while the Fisher computed after trainer.fit follows the model onto the GPU.
    # Folding one into the other used to raise "expected all tensors to be on the
    # same device", and only at the SECOND episode of an EWC chain.
    anchor = ParameterAnchor(lam=1.0, gamma=0.9)
    ones = {n: torch.ones_like(p) for n, p in model.named_parameters()}
    anchor.consolidate(model, ones)

    restored = ParameterAnchor.from_state_dict(anchor.state_dict())
    restored.consolidate(model, ones)  # must not raise

    assert restored.episodes == 2
    torch.testing.assert_close(
        restored.fisher["weight"], 1.9 * torch.ones_like(model.weight)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_consolidate_folds_a_cuda_fisher_into_a_cpu_anchor(model):
    # The exact failure seen on rel-f1: episode 1 saved a CPU anchor, episode 2
    # computed its Fisher on cuda:0, and the fold raised.
    anchor = ParameterAnchor(lam=1.0, gamma=0.9)
    anchor.consolidate(model, {n: torch.ones_like(p) for n, p in model.named_parameters()})
    cpu_anchor = ParameterAnchor.from_state_dict(anchor.state_dict())

    gpu_model = model.cuda()
    gpu_fisher = {n: torch.ones_like(p) for n, p in gpu_model.named_parameters()}
    cpu_anchor.consolidate(gpu_model, gpu_fisher)  # must not raise

    assert cpu_anchor.penalty(gpu_model).item() == pytest.approx(0.0, abs=1e-6)
