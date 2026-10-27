from types import SimpleNamespace

import pytest
import torch
from relbench.base import TaskType

from redelex.nn.train import (
    LightningEntityTaskWrapper,
    SaveModelCallback,
    get_loss,
    get_metrics,
)

from .helpers import (
    UserStaticBinaryTask,
    UserStaticMulticlassTask,
    UserStaticRegressionTask,
)


def test_get_loss_mapping():
    assert isinstance(get_loss(TaskType.BINARY_CLASSIFICATION), torch.nn.BCEWithLogitsLoss)
    assert isinstance(
        get_loss(TaskType.MULTICLASS_CLASSIFICATION), torch.nn.CrossEntropyLoss
    )
    assert isinstance(get_loss(TaskType.REGRESSION), torch.nn.L1Loss)
    with pytest.raises(ValueError, match="unsupported"):
        get_loss(TaskType.RECOMMENDATION)


def test_get_metrics_binary_and_regression():
    metrics, tune, higher = get_metrics(TaskType.BINARY_CLASSIFICATION)
    assert tune in metrics and higher is True

    metrics, tune, higher = get_metrics(TaskType.REGRESSION)
    assert tune in metrics and higher is False


def test_get_metrics_multiclass_requires_num_classes():
    """Regression: multiclass metrics were constructed without num_classes,
    which torchmetrics rejects, so every multiclass wrapper crashed."""
    with pytest.raises(ValueError, match="num_classes"):
        get_metrics(TaskType.MULTICLASS_CLASSIFICATION)

    metrics, tune, higher = get_metrics(TaskType.MULTICLASS_CLASSIFICATION, num_classes=4)
    assert tune in metrics and higher is True
    preds = torch.randn(10, 4).softmax(dim=1)
    target = torch.randint(0, 4, (10,))
    for metric in metrics.values():
        metric.update(preds, target)
        assert torch.isfinite(metric.compute())


def test_wrapper_takes_num_classes_from_task(synthetic_dataset):
    """Regression: the wrapper built multiclass metrics without num_classes,
    which torchmetrics rejects, so every multiclass task crashed here. The task
    declares the count and the wrapper passes it through."""
    task = UserStaticMulticlassTask(synthetic_dataset)
    model = torch.nn.Linear(4, task.num_classes)

    wrapper = LightningEntityTaskWrapper(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        task=task,
    )

    assert not hasattr(wrapper, "num_classes")
    assert wrapper.tune_metric == "macro_roc_auc"
    assert wrapper.higher_is_better is True

    # The metrics were built for three classes, so three-class inputs work.
    preds = torch.randn(6, 3).softmax(dim=1)
    target = torch.tensor([0, 1, 2, 0, 1, 2])
    for metric in wrapper.val_metrics.values():
        metric.update(preds, target)
        assert torch.isfinite(metric.compute())


def test_wrapper_binary_and_regression_tasks(synthetic_dataset):
    for task_cls, tune_metric, higher in [
        (UserStaticBinaryTask, "roc_auc", True),
        (UserStaticRegressionTask, "mae", False),
    ]:
        task = task_cls(synthetic_dataset)
        model = torch.nn.Linear(4, 1)
        wrapper = LightningEntityTaskWrapper(
            model=model,
            optimizer=torch.optim.Adam(model.parameters()),
            task=task,
        )
        assert wrapper.tune_metric == tune_metric
        assert wrapper.higher_is_better is higher


def test_wrapper_rejects_multiclass_task_without_num_classes(synthetic_dataset):
    """A task that cannot report its class count must fail with a clear error."""
    task = UserStaticMulticlassTask(synthetic_dataset)
    model = torch.nn.Linear(4, 3)

    class NoNumClassesTask:
        task_type = TaskType.MULTICLASS_CLASSIFICATION
        entity_table = task.entity_table

    with pytest.raises(ValueError, match="num_classes"):
        LightningEntityTaskWrapper(
            model=model,
            optimizer=torch.optim.Adam(model.parameters()),
            task=NoNumClassesTask(),
        )


def _module(higher_is_better=True):
    return SimpleNamespace(
        model=torch.nn.Linear(1, 1),
        tune_metric="roc_auc",
        higher_is_better=higher_is_better,
    )


def _trainer(metrics, sanity_checking=False, epoch=0):
    return SimpleNamespace(
        callback_metrics=metrics,
        sanity_checking=sanity_checking,
        current_epoch=epoch,
    )


def test_save_model_callback_derives_monitor(tmp_path):
    """Regression: the default monitor 'val_loss_epoch' was never logged by the
    wrapper, so best_model.pt was silently never written."""
    callback = SaveModelCallback(save_dir=str(tmp_path))
    module = _module()

    callback.on_validation_epoch_end(_trainer({"val_roc_auc": torch.tensor(0.6)}), module)
    best = tmp_path / "best_model.pt"
    assert best.exists()

    # A worse score must not overwrite the checkpoint.
    best.unlink()
    callback.on_validation_epoch_end(_trainer({"val_roc_auc": torch.tensor(0.4)}), module)
    assert not best.exists()

    # A better score does.
    callback.on_validation_epoch_end(_trainer({"val_roc_auc": torch.tensor(0.7)}), module)
    assert best.exists()


def test_save_model_callback_ignores_sanity_check(tmp_path):
    callback = SaveModelCallback(save_dir=str(tmp_path))
    module = _module()

    callback.on_validation_epoch_end(
        _trainer({"val_roc_auc": torch.tensor(0.99)}, sanity_checking=True), module
    )
    assert not (tmp_path / "best_model.pt").exists()
    assert callback.best_score is None


def test_save_model_callback_warns_on_missing_monitor(tmp_path):
    callback = SaveModelCallback(save_dir=str(tmp_path), monitor="val_nonexistent")
    module = _module()

    with pytest.warns(UserWarning, match="val_nonexistent"):
        callback.on_validation_epoch_end(_trainer({}), module)
    assert not (tmp_path / "best_model.pt").exists()


def test_save_model_callback_explicit_monitor_min_mode(tmp_path):
    callback = SaveModelCallback(save_dir=str(tmp_path), monitor="val_mae", mode="min")
    module = _module(higher_is_better=False)

    callback.on_validation_epoch_end(_trainer({"val_mae": torch.tensor(1.0)}), module)
    best = tmp_path / "best_model.pt"
    assert best.exists()

    best.unlink()
    callback.on_validation_epoch_end(_trainer({"val_mae": torch.tensor(2.0)}), module)
    assert not best.exists()


@pytest.mark.parametrize(
    "task_cls",
    [UserStaticBinaryTask, UserStaticRegressionTask, UserStaticMulticlassTask],
)
def test_callback_monitor_is_a_metric_the_wrapper_logs(
    task_cls, synthetic_dataset, tmp_path
):
    """The other callback tests drive a stand-in module, so they cannot catch the
    monitored name drifting away from what the wrapper logs -- which is exactly
    the bug that kept best_model.pt from ever being written.
    """
    task = task_cls(synthetic_dataset)
    model = torch.nn.Linear(4, task.num_classes or 1)
    wrapper = LightningEntityTaskWrapper(
        model=model, optimizer=torch.optim.Adam(model.parameters()), task=task
    )

    monitor, mode = SaveModelCallback(save_dir=str(tmp_path))._resolve_monitor(wrapper)

    # `on_validation_epoch_end` logs every val metric under a "val_" prefix.
    assert wrapper.tune_metric in wrapper.val_metrics
    assert monitor == f"val_{wrapper.tune_metric}"
    assert mode == ("max" if wrapper.higher_is_better else "min")


def test_save_model_callback_save_every_epoch(tmp_path):
    callback = SaveModelCallback(save_dir=str(tmp_path), save_every_epoch=True)
    module = _module()

    callback.on_validation_epoch_end(
        _trainer({"val_roc_auc": torch.tensor(0.5)}, epoch=3), module
    )
    assert (tmp_path / "epoch_3_val_roc_auc_0.500.pt").exists()
