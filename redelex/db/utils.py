import numpy as np
import pandas as pd
import sqlalchemy as sa

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
    if driver == "mysql" and not HAS_MYSQL:
        raise ImportError(
            "mysql.connector is not installed. Please install it to use this driver."
        )
    if driver == "pymysql" and not HAS_PYMYSQL:
        raise ImportError("pymysql is not installed. Please install it to use this driver.")

    return f"{dialect}+{driver}://{user}:{password}@{host}:{port}/{database}"


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
    "HAS_PSYCOPG2",
    "HAS_PG8000",
    "HAS_MYSQL",
    "HAS_PYMYSQL",
]
