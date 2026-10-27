from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from relbench.base import Table, TaskType
from sqlalchemy import types as sql_types
from sqlalchemy.dialects.mysql import types as mysql_types
from torch_frame import stype

from redelex.data import guess_column_stype, guess_schema, guess_table_stypes
from redelex.data.semantic_schema import check_predetermined_types

from .helpers import UserStaticBinaryTask


def test_predetermined_bool_is_categorical():
    ser = pd.Series([True, False, True])
    assert check_predetermined_types(ser, None) == stype.categorical


def test_predetermined_float_is_numerical():
    ser = pd.Series([1.5, 2.5, 3.5])
    assert check_predetermined_types(ser, None) == stype.numerical


def test_predetermined_datetime_is_timestamp():
    ser = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02"]))
    assert check_predetermined_types(ser, None) == stype.timestamp


def test_predetermined_longtext_is_text_embedded():
    ser = pd.Series(["lorem", "ipsum"])
    assert check_predetermined_types(ser, mysql_types.LONGTEXT()) == stype.text_embedded


def test_predetermined_plain_int_is_undecided():
    ser = pd.Series([1, 2, 3])
    assert check_predetermined_types(ser, sql_types.Integer()) is None


def test_guess_empty_series_is_none():
    assert guess_column_stype(pd.Series([np.nan, np.nan])) is None


def test_guess_low_cardinality_int_is_categorical():
    ser = pd.Series(np.arange(200) % 4)
    assert guess_column_stype(ser, col_name="status") == stype.categorical


@pytest.mark.parametrize(
    "cardinality,expected", [(1000, True), (1001, False)], ids=["at_limit", "over_limit"]
)
def test_cardinality_limit_is_inclusive(cardinality, expected):
    """The two cardinality bounds must be exact complements, so that every
    column falls in exactly one of the categorical / not-categorical cases."""
    from redelex.data.semantic_schema import (
        MAXIMUM_CARDINALITY_THRESHOLD,
        _is_categorical,
        _is_not_categorical,
    )

    assert MAXIMUM_CARDINALITY_THRESHOLD == 1000
    # Each value is repeated 25 times, putting the distinct-value fraction at
    # 0.04 -- inside both fraction thresholds, so cardinality alone decides.
    ser = pd.Series(np.repeat(np.arange(cardinality), 25))

    assert _is_categorical(ser) is expected
    assert _is_not_categorical(ser) is not expected


def test_guess_high_cardinality_id_is_none():
    ser = pd.Series(np.arange(100))
    assert guess_column_stype(ser, col_name="user_id") is None


def test_guess_high_cardinality_int_is_numerical():
    ser = pd.Series(np.arange(100))
    assert guess_column_stype(ser, col_name="amount") == stype.numerical


def test_guess_mid_cardinality_plural_is_numerical():
    # 10 distinct over 100 rows: between the categorical thresholds, so the
    # decision falls through to the plural-name heuristic.
    ser = pd.Series(np.arange(100) % 10)
    assert guess_column_stype(ser, col_name="goals") == stype.numerical


def test_guess_mid_cardinality_singular_is_categorical():
    ser = pd.Series(np.arange(100) % 10)
    assert guess_column_stype(ser, col_name="city") == stype.categorical


def test_guess_date_strings_are_timestamp():
    ser = pd.Series([f"2020-01-{d:02d}" for d in range(1, 11)])
    assert guess_column_stype(ser, col_name="day_str") == stype.timestamp


def test_guess_free_text_is_text_embedded():
    ser = pd.Series([f"a rather long unique sentence number {i}" for i in range(50)])
    assert guess_column_stype(ser, col_name="description") == stype.text_embedded


def test_guess_all_empty_strings_is_none():
    ser = pd.Series([""] * 10)
    assert guess_column_stype(ser, col_name="note") is None


def test_guess_list_column():
    ser = pd.Series([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert guess_column_stype(ser, col_name="vec") is not None


def test_guess_table_stypes_key_handling():
    df = pd.DataFrame(
        {
            "pk": np.arange(100),
            "fk": np.arange(100) % 5,
            "time": pd.date_range("2020-01-01", periods=100),
            "target": (["yes", "no"] * 50),
            "amount": np.arange(100),
            "row_id": np.arange(100),  # undecided ID column -> dropped
        }
    )
    table = Table(
        df=df,
        fkey_col_to_pkey_table={"fk": "other"},
        pkey_col="pk",
        time_col="time",
    )
    task = SimpleNamespace(target_col="target", task_type=TaskType.BINARY_CLASSIFICATION)

    schema = guess_table_stypes(table, task=task)

    assert schema["pk"] is None
    assert schema["fk"] is None
    assert schema["time"] == stype.timestamp
    assert schema["target"] == stype.categorical
    assert schema["amount"] == stype.numerical
    assert "row_id" not in schema  # ignore_none filters undecided columns

    schema_keep = guess_table_stypes(table, task=task, ignore_none=False)
    assert "row_id" in schema_keep and schema_keep["row_id"] is None


def test_guess_table_stypes_task_types():
    df = pd.DataFrame({"pk": np.arange(10), "target": np.linspace(0, 1, 10)})
    table = Table(df=df, fkey_col_to_pkey_table={}, pkey_col="pk", time_col=None)

    reg = SimpleNamespace(target_col="target", task_type=TaskType.REGRESSION)
    assert guess_table_stypes(table, task=reg)["target"] == stype.numerical

    bad = SimpleNamespace(target_col="target", task_type=TaskType.RECOMMENDATION)
    with pytest.raises(ValueError, match="task type"):
        guess_table_stypes(table, task=bad)


def test_guess_schema_full_db(synthetic_dataset):
    task = UserStaticBinaryTask(synthetic_dataset)
    db = synthetic_dataset.get_db(upto_test_timestamp=False)

    schema = guess_schema(db, task=task)

    assert set(schema) == {"users", "visits"}
    # Task-driven stype only applies on the entity table.
    assert schema["users"]["target_cat"] == stype.categorical
    assert schema["users"]["reg_date"] == stype.timestamp
    assert schema["visits"]["duration"] == stype.numerical
