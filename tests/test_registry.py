"""Offline checks of the CTU dataset and task registrations.

These do not touch the network: registering and instantiating a dataset only
builds the connection URL, the database is downloaded on the first
``get_db()`` call.
"""

import pandas as pd
import pytest
from relbench.base import TaskType

import redelex
from redelex import datasets as redelex_datasets
from redelex import tasks as redelex_tasks
from redelex.datasets import dataset_registry
from redelex.tasks import task_registry

CTU_DATASETS = sorted(n for n in dataset_registry if n.startswith("ctu-"))
CTU_TASKS = [
    (dataset_name, task_name)
    for dataset_name in CTU_DATASETS
    for task_name in sorted(task_registry[dataset_name])
]

SUPPORTED_TASK_TYPES = {
    TaskType.BINARY_CLASSIFICATION,
    TaskType.MULTICLASS_CLASSIFICATION,
    TaskType.REGRESSION,
}


def test_package_exports():
    assert redelex.__version__
    assert set(redelex.__all__) == {"datasets", "tasks", "__version__"}
    for name in redelex.__all__:
        assert hasattr(redelex, name)


def test_dataset_exports_are_importable():
    for name in redelex_datasets.__all__:
        assert hasattr(redelex_datasets, name), f"{name} is exported but missing"
    assert len(set(redelex_datasets.__all__)) == len(redelex_datasets.__all__), (
        "duplicated entries in redelex.datasets.__all__"
    )


def test_task_exports_are_importable():
    for name in redelex_tasks.__all__:
        assert hasattr(redelex_tasks, name), f"{name} is exported but missing"
    assert len(set(redelex_tasks.__all__)) == len(redelex_tasks.__all__), (
        "duplicated entries in redelex.tasks.__all__"
    )


def test_datasets_without_tasks_are_known():
    """The datasets whose only tasks were link prediction have no task left.

    They stay registered so their databases remain usable, but nothing can be
    trained on them until link prediction is implemented. Pinned here so that
    losing a task for any *other* dataset is caught rather than silently
    reducing the smoke test's coverage.
    """
    expected = {
        "ctu-credit",
        "ctu-dunur",
        "ctu-elti",
        "ctu-mooney",
        "ctu-samegen",
        "ctu-satellite",
        "ctu-shakespeare",
    }
    without_tasks = {n for n in CTU_DATASETS if not task_registry.get(n)}
    assert without_tasks == expected


def test_datasets_are_registered_once():
    registered = [
        cls for name, (cls, _, _) in dataset_registry.items() if name.startswith("ctu-")
    ]
    duplicates = {c.__name__ for c in registered if registered.count(c) > 1}
    assert not duplicates, f"datasets registered more than once: {duplicates}"


@pytest.mark.parametrize("dataset_name", CTU_DATASETS)
def test_dataset_instantiates(dataset_name):
    cls, args, kwargs = dataset_registry[dataset_name]
    dataset = cls(*args, **{**kwargs, "cache_dir": None})

    assert isinstance(dataset.val_timestamp, pd.Timestamp)
    assert isinstance(dataset.test_timestamp, pd.Timestamp)
    assert dataset.val_timestamp <= dataset.test_timestamp
    # The password must not leak into logs and tracebacks.
    assert "ctu-relational" not in repr(dataset)


@pytest.mark.parametrize("dataset_name,task_name", CTU_TASKS, ids=lambda v: str(v))
def test_task_instantiates(dataset_name, task_name):
    dataset_cls, dataset_args, dataset_kwargs = dataset_registry[dataset_name]
    dataset = dataset_cls(*dataset_args, **{**dataset_kwargs, "cache_dir": None})

    cls, args, kwargs = task_registry[dataset_name][task_name]
    task = cls(dataset, *args, **{**kwargs, "cache_dir": None})

    assert task.task_type in SUPPORTED_TASK_TYPES, (
        f"{dataset_name}/{task_name}: unsupported task type {task.task_type}"
    )
    assert isinstance(task.entity_table, str) and task.entity_table
    assert isinstance(task.entity_col, str) and task.entity_col
    assert isinstance(task.target_col, str) and task.target_col
