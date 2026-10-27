"""End-to-end smoke tests over every registered CTU dataset and task.

These tests connect to the public CTU relational database at
relational.fel.cvut.cz, download every database and build every task table, so
they are slow and require network access. They are excluded from the default
test run by the ``needs_network`` marker; run them explicitly:

    # everything (downloads ~70 databases, takes a long time)
    pytest -m needs_network

    # a single dataset and its tasks
    pytest -m needs_network -k financial

    # reuse downloads between runs instead of a throw-away directory
    REDELEX_SMOKE_CACHE_DIR=.smoke-cache pytest -m needs_network

    # restrict to a few datasets
    REDELEX_SMOKE_DATASETS=ctu-financial,ctu-seznam pytest -m needs_network
"""

import os
import time

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_datetime64_any_dtype, is_integer_dtype
from relbench.base import TaskType

from redelex.datasets import dataset_registry
from redelex.tasks import task_registry

pytestmark = [pytest.mark.needs_network, pytest.mark.slow]


def _selected_datasets() -> list[str]:
    names = sorted(n for n in dataset_registry if n.startswith("ctu-"))
    subset = os.environ.get("REDELEX_SMOKE_DATASETS")
    if subset:
        wanted = {n.strip() for n in subset.split(",") if n.strip()}
        unknown = wanted - set(names)
        if unknown:
            raise ValueError(f"Unknown datasets in REDELEX_SMOKE_DATASETS: {unknown}")
        names = [n for n in names if n in wanted]
    return names


DATASET_NAMES = _selected_datasets()
TASK_IDS = [
    (dataset_name, task_name)
    for dataset_name in DATASET_NAMES
    for task_name in sorted(task_registry[dataset_name])
]


@pytest.fixture(scope="session")
def smoke_cache_dir(tmp_path_factory) -> str:
    """Cache directory for downloaded databases and task tables.

    Defaults to a throw-away directory so every run really downloads; set
    REDELEX_SMOKE_CACHE_DIR to keep the data between runs.
    """
    from pathlib import Path

    env_dir = os.environ.get("REDELEX_SMOKE_CACHE_DIR")
    if env_dir:
        Path(env_dir).mkdir(parents=True, exist_ok=True)
        return env_dir
    return str(tmp_path_factory.mktemp("ctu-smoke"))


@pytest.fixture(scope="session")
def dataset_factory(smoke_cache_dir):
    """Build (and reuse within the session) dataset objects from the registry.

    Instantiating from the registry directly keeps the cache inside the test
    directory instead of the user's redelex cache.
    """
    built = {}

    def factory(dataset_name: str):
        if dataset_name not in built:
            cls, args, kwargs = dataset_registry[dataset_name]
            kwargs = {**kwargs, "cache_dir": f"{smoke_cache_dir}/{dataset_name}"}
            built[dataset_name] = cls(*args, **kwargs)
        return built[dataset_name]

    return factory


@pytest.mark.parametrize("dataset_name", DATASET_NAMES)
def test_ctu_dataset_downloads(dataset_name, dataset_factory):
    """Download the database and check its structural invariants."""
    dataset = dataset_factory(dataset_name)

    tic = time.time()
    db = dataset.get_db(upto_test_timestamp=False)
    elapsed = time.time() - tic

    assert db.table_dict, f"{dataset_name}: database has no tables"

    total_rows = 0
    for tname, table in db.table_dict.items():
        n_rows = len(table.df)
        total_rows += n_rows
        assert n_rows > 0, f"{dataset_name}.{tname}: table is empty"
        assert not table.df.columns.duplicated().any(), (
            f"{dataset_name}.{tname}: duplicated column names"
        )

        if table.pkey_col is not None:
            pkeys = table.df[table.pkey_col]
            assert (pkeys.values == np.arange(n_rows)).all(), (
                f"{dataset_name}.{tname}: primary key is not consecutive"
            )

        for fkey_col, ref_table in table.fkey_col_to_pkey_table.items():
            assert ref_table in db.table_dict, (
                f"{dataset_name}.{tname}: foreign key {fkey_col} references "
                f"unknown table {ref_table}"
            )
            values = table.df[fkey_col].dropna()
            if len(values) > 0:
                assert values.min() >= 0, (
                    f"{dataset_name}.{tname}.{fkey_col}: negative foreign key"
                )
                assert values.max() < len(db.table_dict[ref_table].df), (
                    f"{dataset_name}.{tname}.{fkey_col}: foreign key out of range "
                    f"of {ref_table}"
                )

        if table.time_col is not None:
            assert table.time_col in table.df.columns, (
                f"{dataset_name}.{tname}: missing time column {table.time_col}"
            )
            assert is_datetime64_any_dtype(table.df[table.time_col]), (
                f"{dataset_name}.{tname}.{table.time_col}: time column is not datetime"
            )

    # Split timestamps must be usable for `db.upto()` comparisons.
    for attr in ["val_timestamp", "test_timestamp"]:
        assert isinstance(getattr(dataset, attr), pd.Timestamp), (
            f"{dataset_name}: {attr} must be a pandas Timestamp"
        )
    assert dataset.val_timestamp <= dataset.test_timestamp, (
        f"{dataset_name}: val_timestamp is after test_timestamp"
    )

    has_time = any(t.time_col is not None for t in db.table_dict.values())
    if has_time:
        assert db.min_timestamp <= db.max_timestamp

    print(
        f"\n{dataset_name}: {len(db.table_dict)} tables, {total_rows} rows, {elapsed:.1f}s"
    )


@pytest.mark.parametrize("dataset_name,task_name", TASK_IDS, ids=lambda v: str(v))
def test_ctu_task_tables(dataset_name, task_name, dataset_factory, smoke_cache_dir):
    """Build the train/val/test tables of every registered task."""
    cls, args, kwargs = task_registry[dataset_name][task_name]
    dataset = dataset_factory(dataset_name)
    kwargs = {
        **kwargs,
        "cache_dir": f"{smoke_cache_dir}/{dataset_name}/tasks/{task_name}",
    }
    task = cls(dataset, *args, **kwargs)

    db = dataset.get_db(upto_test_timestamp=False)
    num_entities = len(db.table_dict[task.entity_table].df)

    tic = time.time()
    entity_ids = {}
    sizes = {}
    observed_labels = set()
    for split in ["train", "val", "test"]:
        table = task.get_table(split, mask_input_cols=False)
        sizes[split] = len(table.df)

        assert len(table.df) > 0, f"{dataset_name}/{task_name}: {split} split is empty"
        assert task.entity_col in table.df.columns
        assert task.target_col in table.df.columns

        ids = table.df[task.entity_col]
        assert ids.notna().all(), (
            f"{dataset_name}/{task_name}: {split} has missing entity ids"
        )
        assert ids.min() >= 0 and ids.max() < num_entities, (
            f"{dataset_name}/{task_name}: {split} entity ids out of range of "
            f"{task.entity_table}"
        )
        entity_ids[split] = set(ids)

        target = table.df[task.target_col]
        if task.task_type == TaskType.BINARY_CLASSIFICATION:
            assert set(target.dropna().unique()) <= {-1.0, 0.0, 1.0}, (
                f"{dataset_name}/{task_name}: unexpected binary labels "
                f"{sorted(set(target.dropna().unique()))[:10]}"
            )
        elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            assert is_integer_dtype(target), (
                f"{dataset_name}/{task_name}: multiclass target is not integer"
            )
            assert target.min() >= -1
            observed_labels |= set(target.unique()) - {-1}
        elif task.task_type == TaskType.REGRESSION:
            assert pd.api.types.is_numeric_dtype(target), (
                f"{dataset_name}/{task_name}: regression target is not numeric"
            )

    elapsed = time.time() - tic

    # Entities are assigned to exactly one split (no leakage between splits).
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = entity_ids[a] & entity_ids[b]
        assert not overlap, (
            f"{dataset_name}/{task_name}: {len(overlap)} entities shared between "
            f"{a} and {b} splits"
        )

    # The default (masked) test table must be buildable and hide the target.
    masked = task.get_table("test")
    assert task.target_col not in masked.df.columns, (
        f"{dataset_name}/{task_name}: masked test table leaks the target column"
    )
    assert len(masked.df) == sizes["test"]

    # `num_classes` is declared per task, so it can disagree with the data. It is
    # what the training metrics are sized from, so a wrong value is a silent
    # error there. The observed count is printed below for the tasks that do not
    # declare one yet.
    num_classes = getattr(task, "num_classes", None)
    if task.task_type == TaskType.MULTICLASS_CLASSIFICATION and num_classes is not None:
        assert observed_labels <= set(range(num_classes)), (
            f"{dataset_name}/{task_name}: declares num_classes={num_classes} but the "
            f"labels reach {max(observed_labels)}"
        )
        assert len(observed_labels) == num_classes, (
            f"{dataset_name}/{task_name}: declares num_classes={num_classes} but the "
            f"splits only contain {len(observed_labels)} distinct labels"
        )

    observed = (
        f", classes = {len(observed_labels)} (declared {num_classes})"
        if task.task_type == TaskType.MULTICLASS_CLASSIFICATION
        else ""
    )
    print(
        f"\n{dataset_name}/{task_name}: {task.task_type.value}, "
        f"train/val/test = {sizes['train']}/{sizes['val']}/{sizes['test']}"
        f"{observed}, {elapsed:.1f}s"
    )
