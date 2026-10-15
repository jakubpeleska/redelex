import time
from types import SimpleNamespace

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import HeteroData

from redelex.nn.train import (
    AnytimeEvalCallback,
    ConvergenceCallback,
    CostMeterCallback,
    LightningEntityTaskWrapper,
    anytime_schedule,
)

from .helpers import UserStaticRegressionTask


class Toy(L.LightningModule):
    """Linear regression whose validation loss is scripted, not computed."""

    def __init__(self, scripted=None):
        super().__init__()
        torch.manual_seed(0)  # identical initial weights in every instance
        self.model = torch.nn.Linear(2, 1)
        self.scripted = list(scripted or [])
        self.snapshots = []
        self.k = 0

    def training_step(self, batch, batch_idx):
        x, y = batch
        return ((self.model(x).squeeze(-1) - y) ** 2).mean()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self._last = ((self.model(x).squeeze(-1) - y) ** 2).mean()

    def on_validation_epoch_end(self):
        value = self.scripted[self.k] if self.scripted else self._last
        self.k += 1
        self.snapshots.append({k: v.clone() for k, v in self.model.state_dict().items()})
        self.log("val_loss", torch.as_tensor(float(value)))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.05)


def _loaders(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g)
    y = x @ torch.tensor([1.5, -2.0]) + 0.1 * torch.randn(n, generator=g)
    train = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)
    val = DataLoader(TensorDataset(x[:16], y[:16]), batch_size=16)
    return train, val


def _fit(module, callbacks, max_steps, val_every=10):
    torch.manual_seed(0)
    train, val = _loaders()
    trainer = L.Trainer(
        max_steps=max_steps,
        val_check_interval=val_every,
        check_val_every_n_epoch=None,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )
    trainer.fit(module, train, val)
    return trainer


def test_convergence_tracks_every_patience_in_one_run(tmp_path):
    #          check: 1    2    3    4    5    6    7    8    9    10
    losses = [5.0, 4.0, 3.0, 3.5, 3.6, 2.9, 3.0, 3.1, 3.2, 3.3, 9, 9, 9]
    module = Toy(losses)
    cb = ConvergenceCallback(str(tmp_path), patiences=(2, 4))

    trainer = _fit(module, [cb], max_steps=500)
    s = cb.summary()["patience"]

    # P=2 gives up after checks 4-5; P=4 waits through the dip at check 6.
    assert s["2"] == {"stop_step": 50, "best_step": 30, "best_value": 3.0, "checks": 5, "hit_cap": False}
    assert s["4"]["stop_step"] == 100 and s["4"]["best_step"] == 60 and not s["4"]["hit_cap"]
    assert trainer.global_step == 100
    p2 = torch.load(tmp_path / "best_P2.pt")
    p4 = torch.load(tmp_path / "best_P4.pt")
    for k in p2:
        assert torch.equal(p2[k], module.snapshots[2][k])  # weights at check 3
        assert torch.equal(p4[k], module.snapshots[5][k])  # weights at check 6
    primary = torch.load(tmp_path / "best_model.pt")
    assert all(torch.equal(primary[k], p2[k]) for k in p2)


def test_convergence_history_replays_every_check(tmp_path):
    module = Toy([3.0, 2.0, 2.5, 2.6, 2.7])
    cb = ConvergenceCallback(str(tmp_path), patiences=(3,))

    _fit(module, [cb], max_steps=500)

    history = cb.summary()["history"]
    assert [s for s, _ in history] == [10, 20, 30, 40, 50]
    assert [v for _, v in history] == pytest.approx([3.0, 2.0, 2.5, 2.6, 2.7])  # logged as float32


def test_convergence_flags_runs_that_hit_the_step_cap(tmp_path):
    module = Toy([10.0 - i for i in range(50)])
    cb = ConvergenceCallback(str(tmp_path), patiences=(3,))

    trainer = _fit(module, [cb], max_steps=40)

    assert trainer.global_step == 40
    assert cb.summary()["patience"]["3"]["hit_cap"] is True
    assert (tmp_path / "best_model.pt").exists()


def test_convergence_requires_the_monitored_value(tmp_path):
    class Silent(Toy):
        def on_validation_epoch_end(self):
            pass

    with pytest.raises(KeyError, match="val_loss"):
        _fit(Silent(), [ConvergenceCallback(str(tmp_path))], max_steps=20)


def test_anytime_schedule():
    on = [s for s in range(0, 9000) if anytime_schedule(s)]
    assert on == [0, 10, 25, 50, 100, 250, 500, 1000, 2000, 4000, 8000]


def _noisy_eval(sleep=0.0):
    def evaluate(model):
        time.sleep(sleep)
        torch.randn(1000)  # consumes torch RNG on purpose
        return {"w0": model.weight[0, 0].item()}

    return evaluate


def test_anytime_eval_does_not_change_training(tmp_path):
    plain = Toy()
    _fit(plain, [], max_steps=60)

    watched = Toy()
    any_cb = AnytimeEvalCallback(_noisy_eval(), base_schedule=(0, 10, 25, 50))
    _fit(watched, [any_cb], max_steps=60)

    for k, v in plain.model.state_dict().items():
        assert torch.equal(v, watched.model.state_dict()[k])
    steps = [r["step"] for r in any_cb.records]
    assert steps == [0, 10, 25, 50, 60]
    assert any_cb.records[-1].get("final") is True


def test_step_zero_sees_the_starting_weights():
    module = Toy()
    with torch.no_grad():
        module.model.weight.fill_(0.123)
    cb = AnytimeEvalCallback(_noisy_eval(), base_schedule=(0,))

    _fit(module, [cb], max_steps=5)

    assert cb.records[0]["step"] == 0
    assert cb.records[0]["w0"] == pytest.approx(0.123)


def _metered(sleep):
    meter = CostMeterCallback()
    cb = AnytimeEvalCallback(_noisy_eval(sleep=sleep), base_schedule=(0, 10, 25, 50), meter=meter)
    _fit(Toy(), [meter, cb], max_steps=60)
    return meter.summary()


def test_cost_meter_real_run_accounts_for_its_wall_clock():
    s = _metered(0.05)

    assert s["steps"] == 60
    assert s["seconds"]["anytime"] >= 5 * 0.05 * 0.9
    accounted = s["train_seconds"] + s["seconds"]["val"] + s["seconds"]["anytime"]
    assert accounted <= s["fit_wall_seconds"] * 1.05


def test_cost_meter_subtracts_measured_work_from_the_data_gap(monkeypatch):
    """Fake clock: the gap between two batches is 8 s, of which 2 s are
    validation and 2 s an anytime evaluation; only 4 s are data time."""
    import redelex.nn.train.callbacks as cbm

    clock = iter([0.0, 1.0, 4.0, 4.0, 6.0, 9.0, 11.0, 12.0, 20.0])
    monkeypatch.setattr(cbm.time, "perf_counter", lambda: next(clock))
    meter = CostMeterCallback()
    trainer = SimpleNamespace(global_step=1)

    meter.on_fit_start(trainer, None)                    # 0
    meter.on_train_batch_start(trainer, None, None, 0)   # 1
    meter.on_train_batch_end(trainer, None, None, None, 0)  # 4  -> 3 s compute
    meter.on_validation_start(trainer, None)             # 4
    meter.on_validation_end(trainer, None)               # 6  -> 2 s val
    with meter.measure("anytime"):                       # 9 .. 11 -> 2 s anytime
        pass
    meter.on_train_batch_start(trainer, None, None, 1)   # 12 -> gap 8 s minus 4 s measured
    meter.on_fit_end(trainer, None)                      # 20

    sec = meter.summary()["seconds"]
    assert sec["train_compute"] == 3.0
    assert sec["val"] == 2.0 and sec["anytime"] == 2.0
    assert sec["train_data"] == 4.0


def test_cost_meter_counts_samples_by_source():
    meter = CostMeterCallback(entity_table="users", source_counts=lambda b: {"new": 3, "old": 5})
    batch = {"users": SimpleNamespace(batch_size=8)}
    trainer = SimpleNamespace(global_step=1)

    meter.on_train_batch_start(trainer, None, batch, 0)
    meter.on_train_batch_end(trainer, None, None, batch, 0)

    assert meter.samples == 8
    assert meter.samples_by_source == {"new": 3, "old": 5}


class _Echo(torch.nn.Module):
    def forward(self, batch, entity_table):
        return batch[entity_table].x


def test_wrapper_val_loss_is_row_weighted(synthetic_dataset):
    task = UserStaticRegressionTask(synthetic_dataset)
    wrapper = LightningEntityTaskWrapper(
        model=_Echo(), optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1), task=task
    )
    preds, targets = [], []
    for n, shift in [(2, 1.0), (6, 3.0)]:
        b = HeteroData()
        b["users"].x = torch.zeros(n, 1)
        b["users"].y = torch.full((n,), shift)
        b["users"].batch_size = n
        wrapper.validation_step(b, 0)
        preds.append(b["users"].x.view(-1))
        targets.append(b["users"].y)

    expected = torch.nn.functional.l1_loss(torch.cat(preds), torch.cat(targets))
    assert wrapper.val_loss.compute().item() == pytest.approx(expected.item())  # 2.5, not 2.0
