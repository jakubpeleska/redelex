import numpy as np
import pandas as pd

from relbench.base import Database


TIMESTAMP_MIN = np.datetime64(pd.Timestamp.min.date())
TIMESTAMP_MAX = np.datetime64(pd.Timestamp.max.date())


def to_unix_time(ser: pd.Series) -> np.ndarray:
    r"""Converts a :class:`pandas.Timestamp` series to UNIX timestamp (in seconds)."""
    assert ser.dtype in [
        np.dtype("datetime64[s]"),
        np.dtype("datetime64[ns]"),
        np.dtype("datetime64[us]"),
        np.dtype("datetime64[ms]"),
    ]
    unix_time = ser.astype("int64").values
    if ser.dtype == np.dtype("datetime64[ns]"):
        unix_time //= 10**9
    elif ser.dtype == np.dtype("datetime64[us]"):
        unix_time //= 10**6
    elif ser.dtype == np.dtype("datetime64[ms]"):
        unix_time //= 10**3
    return unix_time


def convert_timedelta(db: Database):
    """Converts timedelta columns to datetime columns."""

    for table in db.table_dict.values():
        timedeltas = table.df.select_dtypes(include=["timedelta"])
        if not timedeltas.empty:
            timedeltas = pd.Timestamp("1900-01-01") + timedeltas
            table.df[timedeltas.columns] = timedeltas


__all__ = [
    "convert_timedelta",
    "TIMESTAMP_MIN",
    "TIMESTAMP_MAX",
]
