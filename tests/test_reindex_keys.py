import copy
from typing import Dict

import numpy as np
import pandas as pd
import pytest
from relbench.base import Database, Dataset, Table

from redelex.db.utils import reindex_keys


def _relbench2_reindex(db: Database) -> None:
    # relbench 2.1.1 `Database.reindex_pkeys_and_fkeys`, verbatim, as the
    # reference `reindex_keys` has to reproduce.
    index_map_dict: Dict[str, pd.Series] = {}
    for table_name, table in db.table_dict.items():
        if table.pkey_col is not None:
            if table.time_col is not None:
                table.df = table.df.sort_values(table.time_col).reset_index(drop=True)

            ser = table.df[table.pkey_col]

            if ser.nunique() != len(ser):
                raise RuntimeError(
                    f"The primary key '{table.pkey_col}' "
                    f"of table '{table_name}' contains "
                    "duplicated elements"
                )
            arange_ser = pd.RangeIndex(len(ser)).astype("Int64")
            index_map_dict[table_name] = pd.Series(
                index=ser,
                data=arange_ser,
                name="index",
            )
            table.df[table.pkey_col] = arange_ser

    for table in db.table_dict.values():
        for fkey_col, pkey_table_name in table.fkey_col_to_pkey_table.items():
            out = pd.merge(
                table.df[fkey_col],
                index_map_dict[pkey_table_name],
                how="left",
                left_on=fkey_col,
                right_index=True,
            )
            table.df[fkey_col] = out["index"]


def _make_db() -> Database:
    """Keys as DBDataset.make_db leaves them after a customize_db: gaps from
    dropped rows, an unsorted time column with ties and NaT, dangling and
    missing references, and a self-reference."""
    users = pd.DataFrame(
        {
            "__PK__": [0, 1, 2, 4, 5, 7],  # rows 3 and 6 were dropped
            "joined": pd.to_datetime(
                ["2020-03-01", None, "2020-01-01", "2020-03-01", "2020-02-01", None]
            ),
            "referrer": [np.nan, 0.0, 0.0, 7.0, 3.0, 2.0],  # 3 is dangling
            "old_pk": [0, 1, 2, 4, 5, 7],  # tracks the rows through the sort
        },
        index=[0, 1, 2, 4, 5, 7],
    )
    visits = pd.DataFrame(
        {
            "__PK__": np.arange(5),
            "FK_users": [5.0, np.nan, 0.0, 3.0, 7.0],  # 3 is dangling
            "x": np.arange(5) * 1.5,
        }
    )
    # A table without a primary key keeps its rows and index untouched.
    log = pd.DataFrame(
        {"FK_users": [7, 0, 1], "t": pd.to_datetime(["2021", "2020", "2022"])},
        index=[10, 11, 12],
    )
    return Database(
        table_dict={
            "users": Table(
                df=users,
                fkey_col_to_pkey_table={"referrer": "users"},
                pkey_col="__PK__",
                time_col="joined",
            ),
            "visits": Table(
                df=visits,
                fkey_col_to_pkey_table={"FK_users": "users"},
                pkey_col="__PK__",
                time_col=None,
            ),
            "log": Table(
                df=log,
                fkey_col_to_pkey_table={"FK_users": "users"},
                pkey_col=None,
                time_col="t",
            ),
        }
    )


def test_reindex_keys_matches_relbench2():
    expected = _make_db()
    _relbench2_reindex(expected)
    actual = reindex_keys(_make_db())

    for name, table in expected.table_dict.items():
        pd.testing.assert_frame_equal(
            actual.table_dict[name].df, table.df, check_exact=True
        )


def test_reindex_keys_sorts_time_tables_and_renumbers():
    db = reindex_keys(_make_db())
    users = db.table_dict["users"].df

    assert users["__PK__"].dtype == pd.Int64Dtype()
    np.testing.assert_array_equal(users["__PK__"].to_numpy(), np.arange(6))
    # NaT last
    assert users["joined"].iloc[:4].is_monotonic_increasing
    assert users["joined"].iloc[4:].isna().all()
    assert isinstance(users.index, pd.RangeIndex)


def test_reindex_keys_maps_fkeys_and_keeps_dangling_as_na():
    db = reindex_keys(_make_db())
    users = db.table_dict["users"].df
    new_of_old = dict(zip(users["old_pk"], users["__PK__"], strict=True))

    fk = db.table_dict["visits"].df["FK_users"]
    assert fk.dtype == pd.Int64Dtype()
    assert fk.tolist()[0] == new_of_old[5]
    assert pd.isna(fk.iloc[1]) and pd.isna(fk.iloc[3])
    assert fk.iloc[2] == new_of_old[0] and fk.iloc[4] == new_of_old[7]

    # Self-reference: the row with old key 4 was referred by old key 7.
    referrer = users.set_index("__PK__")["referrer"]
    assert referrer[new_of_old[4]] == new_of_old[7]
    assert pd.isna(referrer[new_of_old[5]])  # referred dangling 3


def test_reindex_keys_leaves_tables_without_pkey_in_place():
    db = reindex_keys(_make_db())
    log = db.table_dict["log"].df

    assert log.index.tolist() == [10, 11, 12]
    assert log["t"].tolist() == list(pd.to_datetime(["2021", "2020", "2022"]))


def test_reindex_keys_rejects_duplicate_pkeys():
    db = _make_db()
    db.table_dict["visits"].df["__PK__"] = [0, 1, 1, 2, 3]

    with pytest.raises(RuntimeError, match="duplicated"):
        reindex_keys(db)


def test_reindexed_db_survives_upto():
    """relbench 3's Dataset.get_db cuts the database at test_timestamp and then
    requires consecutive primary keys."""

    class _Dataset(Dataset):
        val_timestamp = pd.Timestamp("2020-01-15")
        test_timestamp = pd.Timestamp("2020-02-15")

        def make_db(self):
            return reindex_keys(_make_db())

    db = _Dataset().get_db()
    assert len(db.table_dict["users"]) == 2  # 2020-01-01 and 2020-02-01

    unsorted = copy.deepcopy(_make_db())

    class _Unsorted(_Dataset):
        def make_db(self):
            return unsorted

    with pytest.raises(RuntimeError, match="consecutively"):
        _Unsorted().get_db()
