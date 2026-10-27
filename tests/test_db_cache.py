import numpy as np
import pandas as pd
from relbench.base import Database, Table

from redelex.db.cache import load_db, load_table, save_db, save_table

from .helpers import SqliteTestDataset, create_sqlite_db


def _table() -> Table:
    df = pd.DataFrame(
        {
            "pk": pd.array([0, 1, 2], dtype="Int64"),
            "fk": pd.array([1, None, 0], dtype="Int64"),
            "t": pd.to_datetime(["2020-01-01", None, "2020-03-01"]),
            "s": pd.array(["a", None, "c"], dtype="string"),
            "x": [1.5, np.nan, 3.0],
        }
    )
    return Table(df=df, fkey_col_to_pkey_table={"fk": "other"}, pkey_col="pk", time_col="t")


def test_table_round_trip(tmp_path):
    table = _table()
    path = tmp_path / "nested" / "train.parquet"
    save_table(table, path)
    out = load_table(path)

    assert out.pkey_col == "pk"
    assert out.time_col == "t"
    assert out.fkey_col_to_pkey_table == {"fk": "other"}
    # As with relbench 2, string columns come back as object columns.
    assert out.df["s"].dtype == object
    pd.testing.assert_frame_equal(out.df, table.df.astype({"s": object}), check_exact=True)


def test_db_round_trip(tmp_path):
    db = Database(
        table_dict={
            "a": _table(),
            "other": Table(pd.DataFrame({"pk": np.arange(3)}), {}, "pk", None),
        }
    )
    save_db(db, tmp_path / "db")
    out = load_db(tmp_path / "db")

    assert sorted(out.table_dict) == ["a", "other"]
    assert sorted(p.name for p in (tmp_path / "db").iterdir()) == [
        "a.parquet",
        "other.parquet",
    ]
    pd.testing.assert_frame_equal(
        out.table_dict["other"].df, db.table_dict["other"].df, check_exact=True
    )


def test_get_db_cold_and_warm_hold_the_same_data(tmp_path):
    url = create_sqlite_db(tmp_path / "source.sqlite")
    cache_dir = tmp_path / "cache"

    cold = SqliteTestDataset(url, cache_dir=str(cache_dir)).get_db(
        upto_test_timestamp=False
    )
    assert any((cache_dir / "db").glob("*.parquet"))
    warm = SqliteTestDataset(url, cache_dir=str(cache_dir)).get_db(
        upto_test_timestamp=False
    )

    assert set(warm.table_dict) == set(cold.table_dict)
    for name, table in cold.table_dict.items():
        # Parquet changes some dtypes (strings, datetime units), not the data.
        pd.testing.assert_frame_equal(warm.table_dict[name].df, table.df, check_dtype=False)
        assert warm.table_dict[name].pkey_col == table.pkey_col
        assert warm.table_dict[name].time_col == table.time_col
        assert warm.table_dict[name].fkey_col_to_pkey_table == table.fkey_col_to_pkey_table


def test_get_db_cache_does_not_touch_source(tmp_path, monkeypatch):
    source = tmp_path / "source.sqlite"
    url = create_sqlite_db(source)
    cache_dir = str(tmp_path / "cache")
    SqliteTestDataset(url, cache_dir=cache_dir).get_db()

    source.unlink()
    dataset = SqliteTestDataset(url, cache_dir=cache_dir)

    def fail():
        raise AssertionError("make_db must not run when the cache exists")

    monkeypatch.setattr(dataset, "make_db", fail)
    db = dataset.get_db()

    assert set(db.table_dict) == {"customer", "product", "orders", "events"}
    # sqlite would have recreated the file on connect.
    assert not source.exists()


def test_get_db_is_cached_in_memory(sqlite_db_url):
    dataset = SqliteTestDataset(sqlite_db_url)

    assert dataset.get_db() is dataset.get_db()
    assert dataset.get_db(upto_test_timestamp=False) is not dataset.get_db()


def test_get_db_reindexes_keys(sqlite_db_url):
    db = SqliteTestDataset(sqlite_db_url).get_db(upto_test_timestamp=False)

    orders = db.table_dict["orders"].df
    assert orders["__PK__"].dtype == pd.Int64Dtype()
    assert orders["FK_customer_cust_id"].dtype == pd.Int64Dtype()
    # The order without a date goes last.
    assert orders["ordered_at"].iloc[:-1].is_monotonic_increasing
    assert pd.isna(orders["ordered_at"].iloc[-1])
