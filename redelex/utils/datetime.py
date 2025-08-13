from typing import Dict, Optional

import numpy as np
import pandas as pd

from torch_frame import stype

from relbench.base import Database, Table


TIMESTAMP_MIN = np.datetime64(pd.Timestamp.min.date())
TIMESTAMP_MAX = np.datetime64(pd.Timestamp.max.date())


def convert_timedelta(db: Database):
    """Converts timedelta columns to datetime columns."""

    for table in db.table_dict.values():
        timedeltas = table.df.select_dtypes(include=["timedelta"])
        if not timedeltas.empty:
            timedeltas = pd.Timestamp("1900-01-01") + timedeltas
            table.df[timedeltas.columns] = timedeltas


def standardize_table_dt(
    table: Table, col_to_stype: Optional[Dict[str, stype]] = None
) -> None:
    """Standartize datetime columns to UNIX timestamp (in datetime[ns] if possible)."""

    if col_to_stype is not None:
        for col, s in col_to_stype.items():
            if s == stype.timestamp:
                table.df[col] = table.df[col].astype(np.dtype("datetime64[ns]"))

    if table.time_col is not None:
        table.df[table.time_col] = table.df[table.time_col].astype(
            np.dtype("datetime64[ns]")
        )


def standardize_db_dt(db: Database, col_to_stype_dict: Dict[str, Dict[str, stype]] = {}):
    """Standartize datetime columns to UNIX timestamp (in datetime[ns] if possible)."""

    for tname, table in db.table_dict.items():
        standardize_table_dt(table, col_to_stype_dict.get(tname, {}))


__all__ = [
    "convert_timedelta",
    "standardize_db_dt",
    "standardize_table_dt",
    "TIMESTAMP_MIN",
    "TIMESTAMP_MAX",
]
