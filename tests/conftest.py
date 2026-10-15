import os

# Must be set before sentence_transformers / huggingface_hub are imported
# anywhere, so an accidental real-embedder construction fails fast instead
# of downloading models.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import random  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from torch_frame import stype  # noqa: E402
from torch_frame.datasets import FakeDataset  # noqa: E402

from .helpers import (  # noqa: E402
    FakeTextEmbedder,
    InMemoryDataset,
    create_sqlite_db,
    make_users_visits_db,
)


@pytest.fixture(autouse=True)
def _seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def sqlite_db_url(tmp_path_factory) -> str:
    db_path = tmp_path_factory.mktemp("db") / "test.sqlite"
    return create_sqlite_db(db_path)


@pytest.fixture()
def synthetic_db():
    return make_users_visits_db()


@pytest.fixture()
def synthetic_dataset():
    # A fresh Database per test: Dataset.get_db is lru_cached and tasks may
    # modify the returned object, so instances must not share state.
    return InMemoryDataset(make_users_visits_db())


@pytest.fixture()
def text_embedder():
    return FakeTextEmbedder()


@pytest.fixture()
def all_stype_dataset():
    ds = FakeDataset(
        num_rows=20,
        with_nan=True,
        stypes=[
            stype.numerical,
            stype.categorical,
            stype.multicategorical,
            stype.timestamp,
        ],
    )
    return ds.materialize()


@pytest.fixture()
def all_stype_tf(all_stype_dataset):
    return all_stype_dataset.tensor_frame


@pytest.fixture()
def num_cat_dataset():
    ds = FakeDataset(num_rows=10, with_nan=False)
    return ds.materialize()


SYNTHETIC_COL_TO_STYPE = {
    "users": {
        "target_cat": stype.categorical,
        "target_num": stype.numerical,
        "age": stype.numerical,
    },
    "visits": {
        "duration": stype.numerical,
    },
}


@pytest.fixture()
def hetero_graph(synthetic_dataset, text_embedder):
    from redelex.data import make_pkey_fkey_graph

    db = synthetic_dataset.get_db(upto_test_timestamp=False)
    col_to_stype = {t: dict(cols) for t, cols in SYNTHETIC_COL_TO_STYPE.items()}
    data, col_stats_dict = make_pkey_fkey_graph(
        db, col_to_stype, text_embedder=text_embedder
    )
    return data, col_stats_dict


STREAM_COL_TO_STYPE = {
    "users": {"age": stype.numerical},
    "events": {"amount": stype.numerical},
}


@pytest.fixture()
def stream_dataset():
    from .helpers import StreamDataset

    return StreamDataset()


@pytest.fixture()
def stream_graph(stream_dataset, text_embedder):
    """Graph of the synthetic event stream, with `time` on both tables."""
    from redelex.data import make_pkey_fkey_graph

    db = stream_dataset.get_db(upto_test_timestamp=False)
    col_to_stype = {t: dict(cols) for t, cols in STREAM_COL_TO_STYPE.items()}
    data, col_stats_dict = make_pkey_fkey_graph(db, col_to_stype, text_embedder=text_embedder)
    return data, col_stats_dict
