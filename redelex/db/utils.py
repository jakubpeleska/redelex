import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import sqlalchemy as sa
from relbench.base import Database

try:
    import psycopg2  # noqa: F401

    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False


try:
    import pg8000  # noqa: F401

    HAS_PG8000 = True
except ImportError:
    HAS_PG8000 = False

try:
    import mysql.connector  # noqa: F401

    HAS_MYSQL = True
except ImportError:
    HAS_MYSQL = False


try:
    import pymysql  # noqa: F401

    HAS_PYMYSQL = True
except ImportError:
    HAS_PYMYSQL = False


SQL_DATE_TYPES = (sa.types.Date, sa.types.DateTime)

SQL_DATE_MAP = {
    sa.types.Date: np.dtype("datetime64[s]"),
    sa.types.DateTime: np.dtype("datetime64[us]"),
    sa.types.Time: np.dtype("timedelta64[us]"),
    sa.types.Interval: np.dtype("timedelta64[us]"),
}

SQL_TO_PANDAS = {
    sa.types.BigInteger: pd.Int64Dtype(),
    sa.types.Boolean: pd.BooleanDtype(),
    sa.types.Date: "object",
    sa.types.DateTime: "object",
    sa.types.Double: pd.Float64Dtype(),
    sa.types.Enum: pd.CategoricalDtype(),
    sa.types.Float: pd.Float64Dtype(),
    sa.types.Integer: pd.Int32Dtype(),
    sa.types.Interval: "object",
    # TODO: Handle binary data
    # sa.types.LargeBinary: "object",
    sa.types.Numeric: pd.Float64Dtype(),
    sa.types.SmallInteger: pd.Int16Dtype(),
    sa.types.String: "string",
    sa.types.Text: "string",
    sa.types.Time: "object",
    sa.types.Unicode: "string",
    sa.types.UnicodeText: "string",
    sa.types.Uuid: "object",
}


def get_db_url(
    dialect: str,
    driver: str,
    user: str,
    password: str,
    host: str,
    port: str,
    database: str,
) -> str:
    """
    Returns the URL for connecting to the remote database in format used by SQLAlchemy.
    For more information, see https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls.

    Args:
        dialect (str): The dialect for the database connection.
        driver (str): The driver for the database connection.
        user (str): The username for the database connection.
        password (str): The password for the database connection.
        host (str): The host address of the remote database.
        port (str): The port number for the database connection.
        database (str): The name of the database.

    Returns:
        str: The URL for connecting to the remote database.
    """
    if driver == "psycopg2" and not HAS_PSYCOPG2:
        raise ImportError(
            "psycopg2 is not installed. Please install it to use this driver."
        )
    if driver == "pg8000" and not HAS_PG8000:
        raise ImportError("pg8000 is not installed. Please install it to use this driver.")
    if driver == "mysqlconnector" and not HAS_MYSQL:
        raise ImportError(
            "mysql.connector is not installed. Please install it to use this driver."
        )
    if driver == "pymysql" and not HAS_PYMYSQL:
        raise ImportError("pymysql is not installed. Please install it to use this driver.")

    # sa.URL.create escapes special characters (e.g. '@' or '/' in passwords).
    url = sa.URL.create(
        drivername=f"{dialect}+{driver}",
        username=user,
        password=password,
        host=host,
        port=int(port) if port is not None else None,
        database=database,
    )
    return url.render_as_string(hide_password=False)


def resolve_column_dtype(
    column: sa.Column,
) -> Tuple[Optional[Union[str, pd.api.extensions.ExtensionDtype]], Optional[type]]:
    """Resolve the pandas dtype and generic SQLAlchemy type of a column.

    Args:
        column (sqlalchemy.Column): The reflected SQLAlchemy column.

    Returns:
        Tuple[Optional[dtype], Optional[type]]: The pandas dtype to read the column
            with and the generic SQLAlchemy type class, or (None, None) when the
            column type is not supported.
    """
    try:
        sql_type = type(column.type.as_generic())
    except NotImplementedError:
        sql_type = None

    dtype = SQL_TO_PANDAS.get(sql_type)

    # Special case for the MySQL YEAR type, which has no generic equivalent.
    if dtype is None and str(column.type) == "YEAR":
        dtype = pd.Int32Dtype()
        sql_type = sa.types.Integer

    return dtype, sql_type


def reindex_fk(
    df_dict: Dict[str, pd.DataFrame],
    src_table: str,
    src_columns: List[str],
    ref_table: str,
    ref_columns: List[str],
) -> Tuple[pd.Series, str]:
    """Map a (possibly composite) foreign key to the referenced table's ``__PK__``.

    Args:
        df_dict (Dict[str, pd.DataFrame]): Dataframes of all tables, each with an
            artificial ``__PK__`` column.
        src_table (str): Name of the table containing the foreign key.
        src_columns (List[str]): The foreign key columns.
        ref_table (str): Name of the referenced table.
        ref_columns (List[str]): The referenced columns.

    Returns:
        Tuple[pd.Series, str]: The re-indexed foreign key values (aligned row-for-row
            with the source table, missing references as NaN) and the new column name.
    """
    fk_name = f"FK_{ref_table}_" + "_".join(src_columns)

    df_src = df_dict[src_table][src_columns]
    df_ref = df_dict[ref_table][[*ref_columns, "__PK__"]]

    if df_ref.duplicated(subset=ref_columns).any():
        warnings.warn(
            f"Referenced columns {ref_columns} of table '{ref_table}' are not "
            f"unique; foreign key '{fk_name}' of table '{src_table}' is resolved "
            "to the first matching row.",
            stacklevel=2,
        )
        df_ref = df_ref.drop_duplicates(subset=ref_columns, keep="first")

    fk_col = df_src.merge(
        df_ref,
        how="left",
        left_on=src_columns,
        right_on=ref_columns,
    )["__PK__"]

    # pandas merge matches NaN keys to NaN keys; a missing key must stay dangling.
    na_mask = df_src.isna().any(axis=1).to_numpy()
    if na_mask.any():
        fk_col = fk_col.mask(na_mask)

    if len(fk_col) != len(df_src):
        raise RuntimeError(
            f"Re-indexing foreign key '{fk_name}' of table '{src_table}' changed "
            f"the row count ({len(df_src)} -> {len(fk_col)})."
        )

    return fk_col, fk_name


def reindex_keys(db: Database) -> Database:
    """Sort time-indexed tables by time and re-index their primary and foreign keys.

    Primary keys are replaced by consecutive indices ``0..n-1`` in row order and
    foreign keys are mapped onto them; missing references become ``<NA>``. Tables
    with a primary key and a time column are sorted by time first, so that rows cut
    by ``Database.upto`` keep their keys.

    Args:
        db (Database): The database to re-index in place.

    Returns:
        Database: The re-indexed database.
    """
    # A port of relbench 2.1 `Database.reindex_pkeys_and_fkeys`, which relbench 3
    # dropped while its `Dataset.get_db` still requires the result. The sort call is
    # kept as it was so that row order (and thus the static task splits) does not
    # change; numpy's default sort is not stable, ties follow its implementation.
    pkey_dict: Dict[str, pd.DataFrame] = {}
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
            # Old key under a temporary name, the new one as `__PK__`, the
            # layout `reindex_fk` resolves references against.
            pkey_dict[table_name] = pd.DataFrame(
                {"__old_key__": ser.to_numpy(), "__PK__": arange_ser}
            )
            table.df[table.pkey_col] = arange_ser

    for table in db.table_dict.values():
        for fkey_col, pkey_table_name in table.fkey_col_to_pkey_table.items():
            # Separate dict keys, a self-referencing foreign key points at its
            # own table.
            fk_col, _ = reindex_fk(
                {"src": table.df, "ref": pkey_dict[pkey_table_name]},
                "src",
                [fkey_col],
                "ref",
                ["__old_key__"],
            )
            # `reindex_fk` returns a fresh index; the table may keep its own.
            table.df[fkey_col] = fk_col.set_axis(table.df.index)

    return db


def get_db_connection(connection_url: str) -> sa.Connection:
    """
    Create a new SQLAlchemy Connection instance to the remote database.
    Don't forget to close the Connection after you are done using it!

    Args:
        connection_url (str): The URL for connecting to the remote database.
            Format is dialect+driver://username:password@host:port/database

    Returns:
        Connection: The SQLAlchemy Connection instance to the remote database.
    """
    return sa.Connection(sa.create_engine(connection_url))


__all__ = [
    "SQL_DATE_MAP",
    "SQL_DATE_TYPES",
    "SQL_TO_PANDAS",
    "get_db_url",
    "get_db_connection",
    "resolve_column_dtype",
    "reindex_fk",
    "reindex_keys",
    "HAS_PSYCOPG2",
    "HAS_PG8000",
    "HAS_MYSQL",
    "HAS_PYMYSQL",
]
