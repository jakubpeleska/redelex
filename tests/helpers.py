"""Shared offline test helpers: fake embedder, synthetic databases and tasks.

Everything here must work without network access: no remote MariaDB, no
model downloads. DB-layer tests use a local sqlite file, task and graph
tests use an in-memory relbench dataset.
"""

import numpy as np
import pandas as pd
import sqlalchemy as sa
from relbench.base import Database, Dataset, EntityTask, Table, TaskType
from torch_frame.testing.text_embedder import HashTextEmbedder

from redelex.datasets.db_dataset import DBDataset
from redelex.tasks.task_impute import ImputeEntityStaticTask, ImputeEntityTemporalTask


class FakeTextEmbedder:
    """Offline stand-in for redelex.data.TextEmbedder (hash-based, no downloads)."""

    embedding_dim = 16

    def __init__(self):
        self._embedder = HashTextEmbedder(self.embedding_dim)

    def __call__(self, sentences: list[str]) -> "torch.Tensor":  # noqa: F821
        return self._embedder(list(sentences))


# ---------------------------------------------------------------------------
# sqlite database for redelex/db and redelex/datasets tests
# ---------------------------------------------------------------------------

SQLITE_DDL = [
    """
    CREATE TABLE customer (
        id INTEGER PRIMARY KEY,
        name TEXT,
        age INTEGER,
        signup_date DATETIME
    )
    """,
    """
    CREATE TABLE product (
        region TEXT,
        code INTEGER,
        price REAL,
        PRIMARY KEY (region, code)
    )
    """,
    """
    CREATE TABLE orders (
        oid INTEGER PRIMARY KEY,
        cust_id INTEGER REFERENCES customer(id),
        region TEXT,
        code INTEGER,
        ordered_at DATETIME,
        FOREIGN KEY (region, code) REFERENCES product(region, code)
    )
    """,
    """
    CREATE TABLE events (
        eid INTEGER PRIMARY KEY,
        cust_name TEXT REFERENCES customer(name),
        happened_at DATETIME
    )
    """,
]

SQLITE_ROWS = {
    "customer": [
        # Two customers intentionally share the name "Alice" (non-unique
        # referenced column for the events.cust_name foreign key).
        (1, "Alice", 30, "2020-01-01 10:00:00"),
        (2, "Alice", 41, "2020-02-01 10:00:00"),
        (3, "Bob", 25, "2020-03-01 10:00:00"),
        (4, "Carol", 52, None),
        (5, "Dan", 33, "2020-05-01 10:00:00"),
        (6, "Eve", 28, "2020-06-01 10:00:00"),
    ],
    "product": [
        ("EU", 1, 9.99),
        ("EU", 2, 19.99),
        ("US", 1, 4.99),
    ],
    "orders": [
        # cust_id=99 is dangling; ("AS", 9) is a dangling composite key.
        (1, 1, "EU", 1, "2020-01-05 12:00:00"),
        (2, 2, "EU", 2, "2020-02-05 12:00:00"),
        (3, 3, "US", 1, "2020-03-05 12:00:00"),
        (4, 99, "EU", 1, "2020-04-05 12:00:00"),
        (5, 5, "AS", 9, "2020-05-05 12:00:00"),
        (6, 6, "EU", 2, None),
    ],
    "events": [
        (1, "Alice", "2020-01-10 08:00:00"),
        (2, "Bob", "2020-03-10 08:00:00"),
        (3, "Eve", "2020-06-10 08:00:00"),
        (4, None, "2020-07-10 08:00:00"),
    ],
}


def create_sqlite_db(db_path) -> str:
    """Create the test sqlite database file and return its SQLAlchemy URL."""
    url = f"sqlite:///{db_path}"
    engine = sa.create_engine(url)
    with engine.begin() as conn:
        for ddl in SQLITE_DDL:
            conn.execute(sa.text(ddl))
        for table, rows in SQLITE_ROWS.items():
            placeholders = ", ".join(f":p{i}" for i in range(len(rows[0])))
            for row in rows:
                conn.execute(
                    sa.text(f"INSERT INTO {table} VALUES ({placeholders})"),
                    {f"p{i}": v for i, v in enumerate(row)},
                )
    engine.dispose()
    return url


class SqliteTestDataset(DBDataset):
    """DBDataset over the local sqlite test database."""

    val_timestamp = pd.Timestamp("2020-05-01")
    test_timestamp = pd.Timestamp("2020-06-15")

    def __init__(self, remote_url: str, **kwargs):
        kwargs.setdefault("time_col_dict", {"orders": "ordered_at"})
        super().__init__(remote_url=remote_url, **kwargs)


# ---------------------------------------------------------------------------
# In-memory relbench dataset for task, graph and loader tests
# ---------------------------------------------------------------------------

VAL_TIMESTAMP = pd.Timestamp("2020-01-01")
TEST_TIMESTAMP = pd.Timestamp("2020-07-01")

N_USERS = 20
N_VISITS = 30
NAN_TARGET_CAT_ROWS = [3, 8, 15]
NAN_TARGET_NUM_ROWS = [5, 12]
DANGLING_VISIT_ROW = 7


def make_users_visits_db() -> Database:
    """Two-table database: `users` (entity table with targets) and `visits`.

    Rows are pre-sorted by their time columns with arange primary keys, so
    relbench's reindex_pkeys_and_fkeys keeps row order stable.
    """
    reg_dates = pd.date_range("2019-01-01", periods=N_USERS, freq="MS")
    target_cat = pd.Series(["yes", "no"] * (N_USERS // 2), dtype=object)
    target_cat.iloc[NAN_TARGET_CAT_ROWS] = np.nan
    # Three classes, so tests can tell an inferred class count from a hardcoded one.
    target_multi = pd.Series(["low", "mid", "high"] * (N_USERS // 3 + 1), dtype=object)
    target_multi = target_multi.iloc[:N_USERS].reset_index(drop=True)
    target_multi.iloc[NAN_TARGET_CAT_ROWS] = np.nan
    target_num = pd.Series(np.linspace(0.0, 9.5, N_USERS))
    target_num.iloc[NAN_TARGET_NUM_ROWS] = np.nan

    users = pd.DataFrame(
        {
            "__PK__": np.arange(N_USERS),
            "reg_date": reg_dates,
            "target_cat": target_cat,
            "target_multi": target_multi,
            "target_num": target_num,
            "age": np.arange(18, 18 + N_USERS),
        }
    )

    visit_dates = pd.date_range("2019-01-15", periods=N_VISITS, freq="3W")
    fk_users = np.arange(N_VISITS) % N_USERS
    fk_users[DANGLING_VISIT_ROW] = 999  # dangling foreign key
    visits = pd.DataFrame(
        {
            "__PK__": np.arange(N_VISITS),
            "FK_users": fk_users,
            "visit_date": visit_dates,
            "duration": np.linspace(1.0, 30.0, N_VISITS),
        }
    )

    return Database(
        table_dict={
            "users": Table(
                df=users,
                fkey_col_to_pkey_table={},
                pkey_col="__PK__",
                time_col="reg_date",
            ),
            "visits": Table(
                df=visits,
                fkey_col_to_pkey_table={"FK_users": "users"},
                pkey_col="__PK__",
                time_col="visit_date",
            ),
        }
    )


class InMemoryDataset(Dataset):
    """relbench Dataset over an in-memory Database, no cache directory."""

    val_timestamp = VAL_TIMESTAMP
    test_timestamp = TEST_TIMESTAMP

    def __init__(self, db: Database):
        self._db = db
        super().__init__(cache_dir=None)

    def make_db(self) -> Database:
        return self._db


class UserStaticBinaryTask(ImputeEntityStaticTask):
    entity_col = "__PK__"
    entity_table = "users"
    target_col = "target_cat"
    task_type = TaskType.BINARY_CLASSIFICATION


class UserStaticRegressionTask(ImputeEntityStaticTask):
    entity_col = "__PK__"
    entity_table = "users"
    target_col = "target_num"
    task_type = TaskType.REGRESSION


class UserStaticMulticlassTask(ImputeEntityStaticTask):
    """Multiclass task over a three-category target."""

    entity_col = "__PK__"
    entity_table = "users"
    target_col = "target_multi"
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    # Declared per task; nothing derives it from the data.
    num_classes = 3


class UserTemporalBinaryTask(ImputeEntityTemporalTask):
    entity_col = "__PK__"
    entity_table = "users"
    target_col = "target_cat"
    task_type = TaskType.BINARY_CLASSIFICATION
    timedelta = pd.Timedelta(days=30)


# ---------------------------------------------------------------------------
# Synthetic event stream for continual-learning tests: a forecasting task with
# its own make_table, enough entities for an entity holdout, and enough time
# for several update windows.
# ---------------------------------------------------------------------------

STREAM_START = pd.Timestamp("2019-01-01")
STREAM_VAL_TIMESTAMP = pd.Timestamp("2021-07-01")
STREAM_TEST_TIMESTAMP = pd.Timestamp("2022-01-01")
STREAM_END = pd.Timestamp("2022-07-01")
STREAM_N_USERS = 300
STREAM_N_EVENTS = 6000


def make_stream_db(seed: int = 0) -> Database:
    """`users` (entity table, staggered sign-ups) and `events` (one stream).

    Each user's events start at sign-up; event rates differ per user, so churn
    and counts are learnable from history. One numerical cell is missing.
    """
    rng = np.random.default_rng(seed)
    span_days = (STREAM_END - STREAM_START).days
    signup = STREAM_START + pd.to_timedelta(
        np.sort(rng.integers(0, int(span_days * 0.8), STREAM_N_USERS)), unit="D"
    )
    age = rng.integers(18, 70, STREAM_N_USERS).astype(float)
    age[7] = np.nan
    users = pd.DataFrame({"__PK__": np.arange(STREAM_N_USERS), "signup": signup, "age": age})

    rate = rng.gamma(2.0, 1.0, STREAM_N_USERS)
    owner = rng.choice(STREAM_N_USERS, size=STREAM_N_EVENTS, p=rate / rate.sum())
    life = (STREAM_END - signup[owner]).days.to_numpy()
    when = signup[owner] + pd.to_timedelta((rng.random(STREAM_N_EVENTS) * life).astype(int), unit="D")
    order = np.argsort(when.to_numpy(), kind="stable")
    events = pd.DataFrame(
        {
            "__PK__": np.arange(STREAM_N_EVENTS),
            "FK_users": owner[order],
            "event_time": when[order].to_numpy(),
            "amount": rng.gamma(2.0, 10.0, STREAM_N_EVENTS)[order],
        }
    )
    return Database(
        table_dict={
            "users": Table(df=users, fkey_col_to_pkey_table={}, pkey_col="__PK__", time_col="signup"),
            "events": Table(
                df=events,
                fkey_col_to_pkey_table={"FK_users": "users"},
                pkey_col="__PK__",
                time_col="event_time",
            ),
        }
    )


class StreamDataset(Dataset):
    val_timestamp = STREAM_VAL_TIMESTAMP
    test_timestamp = STREAM_TEST_TIMESTAMP

    def __init__(self, seed: int = 0):
        self._seed = seed
        super().__init__(cache_dir=None)

    def make_db(self) -> Database:
        return make_stream_db(self._seed)


def _active_users(db: Database, timestamps) -> pd.DataFrame:
    """(timestamp, user) pairs for users with an event in the 90 days up to t."""
    events = db.table_dict["events"].df
    rows = []
    for t in pd.Series(timestamps):
        recent = events[(events.event_time > t - pd.Timedelta(days=90)) & (events.event_time <= t)]
        for u in np.unique(recent.FK_users.to_numpy()):
            rows.append((t, u))
    return pd.DataFrame(rows, columns=["timestamp", "user"])


def _future_counts(db: Database, pairs: pd.DataFrame, delta: pd.Timedelta) -> np.ndarray:
    events = db.table_dict["events"].df
    counts = np.zeros(len(pairs), dtype=float)
    for t, idx in pairs.groupby("timestamp").groups.items():
        window = events[(events.event_time > t) & (events.event_time <= t + delta)]
        per_user = window.FK_users.value_counts()
        counts[np.asarray(pairs.index.get_indexer(idx))] = (
            pairs.loc[idx, "user"].map(per_user).fillna(0).to_numpy()
        )
    return counts


class StreamChurnTask(EntityTask):
    """Will an active user have no event in the next 30 days?"""

    task_type = TaskType.BINARY_CLASSIFICATION
    entity_col = "user"
    entity_table = "users"
    time_col = "timestamp"
    target_col = "churn"
    timedelta = pd.Timedelta(days=30)
    num_eval_timestamps = 3
    metrics = []

    def make_table(self, db: Database, timestamps) -> Table:
        pairs = _active_users(db, timestamps)
        pairs["churn"] = (_future_counts(db, pairs, self.timedelta) == 0).astype(int)
        # Shuffled on purpose: relbench does not promise a row order.
        df = pairs.sample(frac=1.0, random_state=1).reset_index(drop=True)
        return Table(
            df=df,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )


class StreamCountTask(EntityTask):
    """How many events will an active user have in the next 30 days?"""

    task_type = TaskType.REGRESSION
    entity_col = "user"
    entity_table = "users"
    time_col = "timestamp"
    target_col = "count"
    timedelta = pd.Timedelta(days=30)
    num_eval_timestamps = 3
    metrics = []

    def make_table(self, db: Database, timestamps) -> Table:
        pairs = _active_users(db, timestamps)
        pairs["count"] = _future_counts(db, pairs, self.timedelta)
        return Table(
            df=pairs,
            fkey_col_to_pkey_table={self.entity_col: self.entity_table},
            pkey_col=None,
            time_col=self.time_col,
        )
