from typing import Dict, Optional

import re
import warnings

from sqlalchemy.dialects.mysql import types as mysql_types
from sqlalchemy import types as sql_types

import numpy as np

import pandas as pd
from pandas.api import types as pd_types
from dateutil.parser import ParserError

import inflect

from torch_frame import stype
from torch_frame.utils import infer_series_stype

from relbench.base import BaseTask, Database, Table, TaskType

ID_NAME_REGEX = re.compile(
    r"_id$|^id_|_id_|Id$|Id[^a-z]|[Ii]dentifier|IDENTIFIER|ID[^a-zA-Z]|ID$|[guGU]uid[^a-z]|[guGU]uid$|[GU]UID[^a-zA-Z]|[GU]UID$"
)

COMMON_NUMERIC_COLUMN_NAME_REGEX = re.compile(
    r"balance|amount|size|duration|frequency|count|cnt|votes|score|number|age|year|month|day",
    re.IGNORECASE,
)

FRACTION_DISTINCT_NONNULL_GUARANTEED_THRESHOLD = 0.05
"""
The fraction of distinct values to total count of non-null values,
which decides (in some situations) that type must be categorical.
If the fraction is below this threshold, marks the column as categorical.
"""

FRACTION_DISTINCT_NONNULL_IGNORE_THRESHOLD = 0.2
"""
The fraction of distinct values to total count of non-null values,
which decides (in some situations) that type cannot be categorical.
If the fraction exceeds this threshold, marks the column as something other than categorical.
"""

MAXIMUM_CARDINALITY_THRESHOLD = 1000
"""
Maximum number of cardinality that a categorical column can have.
This is present to limit incorrectly classified categorical columns.
"""

POSSIBLE_TIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d",
    "%Y/%m/%d",
]
"""

"""

pd.to_datetime


def check_predetermined_types(
    ser: pd.Series, sql_type: Optional[sql_types.TypeEngine]
) -> Optional[stype]:
    if sql_type is None:
        sql_type = sql_types.NullType()
    sql_type_generic = sql_type.as_generic(allow_nulltype=True)

    if isinstance(sql_type, (mysql_types.LONGTEXT, mysql_types.MEDIUMTEXT)) or isinstance(
        sql_type_generic, (sql_types.Unicode)
    ):
        return stype.text_embedded

    if isinstance(sql_type_generic, (sql_types.Boolean)) or pd_types.is_bool_dtype(ser):
        return stype.categorical

    if isinstance(sql_type_generic, (sql_types.Numeric)) or pd_types.is_float_dtype(ser):
        return stype.numerical

    if isinstance(
        sql_type_generic, (sql_types.Date, sql_types.DateTime, sql_types.Time)
    ) or pd_types.is_datetime64_any_dtype(ser):
        return stype.timestamp

    return None


def guess_column_stype(
    ser: pd.Series, col_name: str = "", sql_type: Optional[sql_types.TypeEngine] = None
) -> stype:
    ser = ser.dropna()
    if ser.empty:
        return None

    if sql_type is None:
        sql_type = sql_types.NullType()
    sql_generic_type = sql_type.as_generic(allow_nulltype=True)

    # Handle list data with torch frame infer
    if isinstance(ser.iloc[0], (list, np.ndarray)):
        if isinstance(ser.iloc[0], np.ndarray):
            ser = ser.apply(np.ndarray.tolist)
        return infer_series_stype(ser)

    predeter_type = check_predetermined_types(ser, sql_type)
    if predeter_type is not None:
        return predeter_type

    # check if cardinality to total count ratio is below categorical threshold
    if _is_categorical(ser):
        return stype.categorical

    if isinstance(sql_generic_type, sql_types.Integer) or pd_types.is_numeric_dtype(ser):
        # check if there are too many distinct values compared to total
        if _is_not_categorical(ser):
            if _is_id_name(col_name):
                return None
            return stype.numerical

        # try matching based on common regex names
        if COMMON_NUMERIC_COLUMN_NAME_REGEX.search(col_name):
            return stype.numerical

        # check if the column name is plural - then it is probably a count
        if _is_plural(col_name):
            return stype.numerical

        return stype.categorical

    elif isinstance(
        sql_generic_type, (sql_types.String, sql_types.Text)
    ) or pd_types.is_string_dtype(ser):
        # if the column contains only empty strings, we can ignore it
        if (ser == "").all():
            return None

        # check if the values fit some of the time formats
        if _is_timestamp(ser):
            return stype.timestamp

        # check if there are too many distinct values compared to total
        if _is_not_categorical(ser):
            if _is_id_name(col_name):
                return None
            return stype.text_embedded

        return stype.categorical

    # no decision - omit
    return None


def guess_table_stypes(
    table: Table,
    table_schema: Optional[Dict[str, sql_types.TypeEngine]] = {},
    task: Optional[BaseTask] = None,
    ignore_none: bool = True,
) -> Dict[str, stype]:
    """
    Guess the stypes of columns in the table.

    Contains additional logic for foreign keys and filtering based on constructor input.
    """

    schema: Dict[str, stype] = {}

    for col in table.df.columns:
        if table.pkey_col is not None and col == table.pkey_col:
            schema[col] = None
            continue

        if table.time_col is not None and col == table.time_col:
            schema[col] = stype.timestamp
            continue

        if col in table.fkey_col_to_pkey_table:
            schema[col] = None
            continue

        if task is not None:
            if task.task_type == TaskType.BINARY_CLASSIFICATION:
                schema[col] = stype.categorical
            elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                schema[col] = stype.categorical
            elif task.task_type == TaskType.REGRESSION:
                schema[col] = stype.numerical
            elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
                schema[col] = stype.multicategorical
            else:
                raise ValueError(f"Unknown task type {task.task_type.name}")
            continue

        guess = guess_column_stype(
            table.df[col], col_name=col, sql_type=table_schema.get(col, None)
        )
        if ignore_none and guess is None:
            continue

        schema[col] = guess
    return schema


def guess_schema(
    db: Database, sql_schema: Optional[Dict[str, Dict[str, sql_types.TypeEngine]]] = {}
) -> Dict[str, Dict[str, stype]]:
    """Locate all database tables and all columns and run :py:method:`guess_column_type` for all of them.

    Returns the result as a :py:class:`Schema`.
    """
    schema = {}

    for table_name, table in db.table_dict.items():
        schema[table_name] = guess_table_stypes(
            table, table_schema=sql_schema.get(table_name, {})
        )

    return schema


def _is_not_categorical(ser: pd.Series) -> bool:
    cardinality = ser.unique().size
    n_nonnull = ser.count()

    return (
        cardinality / n_nonnull > FRACTION_DISTINCT_NONNULL_IGNORE_THRESHOLD
        or cardinality > MAXIMUM_CARDINALITY_THRESHOLD
    )


def _is_categorical(ser: pd.Series) -> bool:
    cardinality = ser.unique().size
    n_nonnull = ser.count()

    return (
        cardinality / n_nonnull <= FRACTION_DISTINCT_NONNULL_GUARANTEED_THRESHOLD
        and cardinality < MAXIMUM_CARDINALITY_THRESHOLD
    )


def _is_id_name(col_name: str) -> bool:
    return ID_NAME_REGEX.search(col_name) is not None


def _is_plural(s: str) -> bool:
    p = inflect.engine()
    return p.singular_noun(s) is not False


def _is_timestamp(ser: pd.Series) -> bool:
    r"""Check if a series is a timestamp.

    Taken from torch_frame
    https://github.com/pyg-team/pytorch-frame/blob/1456fab68ad291e8fe0572716704c69f36409bf6/torch_frame/utils/infer_stype.py#L21
    """
    for time_format in POSSIBLE_TIME_FORMATS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pd.to_datetime(ser, format=time_format)
            return True
        except (ValueError, ParserError, TypeError):
            pass
    return False


__all__ = [
    "check_predetermined_types",
    "guess_column_stype",
    "guess_table_stypes",
    "guess_schema",
]
