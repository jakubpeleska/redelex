import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import sqlalchemy as sa
from relbench.base import Database, Dataset, Table
from tqdm.std import tqdm

from redelex.db import DBInspector, ForeignKey
from redelex.db.cache import load_db, save_db
from redelex.db.utils import (
    SQL_DATE_MAP,
    SQL_DATE_TYPES,
    get_db_connection,
    get_db_url,
    reindex_fk,
    reindex_keys,
    resolve_column_dtype,
)

__all__ = ["DBDataset"]


class DBDataset(Dataset):
    """
    A dataset that is created from a remote relational database.

    Attributes:
        remote_url (str): The URL for connecting to the remote database.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        remote_url: Optional[str] = None,
        dialect: Optional[str] = None,
        driver: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[Union[str, int]] = None,
        database: Optional[str] = None,
        time_col_dict: Optional[Dict[str, str]] = None,
        keep_original_keys: bool = False,
        keep_original_compound_keys: bool = True,
    ):
        """Create a database dataset object.

        Args:
            cache_dir (str, optional): The directory to cache the dataset. Defaults to None.
            remote_url (str, optional): The URL for connecting to the remote database in SQLAlchemy format. \
                For more information, see https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls. \
                If not defined, all following parameters have to be specified. Defaults to None.
            dialect (str, optional): The dialect for the database connection. Defaults to None.
            driver (str, optional): The driver for the database connection. Defaults to None.
            user (str, optional): The username for the database connection. Defaults to None.
            password (str, optional): The password for the database connection. Defaults to None.
            host (str, optional): The host address of the remote database. Defaults to None.
            port (str, optional): The port number for the database connection. Defaults to None.
            database (str, optional): The name of the database. Defaults to None.
            time_col_dict (Dict[str, str], optional): A dictionary mapping table names to time columns. Defaults to None.
            keep_original_keys (bool, optional): Whether to keep original primary and foreign keys \
                after duplication during re-indexing. This is useful when the keys contain information \
                beyond just their relationship to other rows. Defaults to False.
            keep_original_compound_keys (bool, optional): Whether to keep original compound primary \
                and foreign keys as they often contain useful data. Defaults to True.
        """

        self.remote_url = (
            remote_url
            if remote_url is not None
            else get_db_url(dialect, driver, user, password, host, port, database)
        )

        self.time_col_dict = time_col_dict if time_col_dict is not None else {}

        self.keep_original_keys = keep_original_keys
        self.keep_original_compound_keys = keep_original_compound_keys

        self.cache_dir = cache_dir

    def __repr__(self) -> str:
        # Render via sa.URL to hide the password in logs and tracebacks.
        url = sa.engine.make_url(self.remote_url)
        return f"{self.__class__.__name__}(remote_url={url})"

    # Cached per instance like relbench 2's Dataset.get_db; the dataset is kept
    # alive by the cache on purpose.
    @lru_cache(maxsize=None)  # noqa: B019
    def get_db(self, upto_test_timestamp: bool = True) -> Database:
        r"""Return the database object.

        The returned database object is cached in memory. If ``cache_dir`` is set,
        it is also cached on disk in ``<cache_dir>/db``: the first call downloads
        and processes the database, later calls load the cached parquet files.

        Args:
            upto_test_timestamp: If True, only return rows upto test_timestamp.

        Returns:
            Database: The database object.

        `upto_test_timestamp` is True by default to prevent test leakage.
        """
        # A port of relbench 2.1 `Dataset.get_db`. relbench 3 reduced it to
        # make_db -> upto -> validate_and_correct_db, without the caching and
        # without re-indexing the keys, which `upto` relies on.
        db_path = f"{self.cache_dir}/db"
        if self.cache_dir and Path(db_path).exists() and any(Path(db_path).iterdir()):
            print(f"Loading Database object from {db_path}...")
            tic = time.time()
            db = load_db(db_path)
            toc = time.time()
            print(f"Done in {toc - tic:.2f} seconds.")

        else:
            print("Making Database object from scratch...")
            tic = time.time()
            db = self.make_db()
            reindex_keys(db)
            toc = time.time()
            print(f"Done in {toc - tic:.2f} seconds.")

            if self.cache_dir:
                print(f"Caching Database object to {db_path}...")
                tic = time.time()
                save_db(db, db_path)
                toc = time.time()
                print(f"Done in {toc - tic:.2f} seconds.")

        if upto_test_timestamp:
            db = db.upto(self.test_timestamp)

        self.validate_and_correct_db(db)

        return db

    def customize_db(self, db: Database) -> Database:
        """
        Override this method to add custom modifications to the database object.
        Function is called after the database is created and before the original
        primary and foreign keys are removed. The default implementation returns
        the database unchanged.

        Args:
            db (Database): The database object to customize.

        Returns:
            Database: The customized database object.
        """
        return db

    def make_db(self) -> Database:
        """
        Create a Database instance from the remote database.

        Returns:
            Database: The Database instance.
        """
        remote_con = get_db_connection(self.remote_url)
        try:
            inspector = DBInspector(remote_con)

            remote_md = sa.MetaData()
            remote_md.reflect(bind=inspector.engine)

            table_names = inspector.get_tables()

            df_dict: Dict[str, pd.DataFrame] = {}
            fk_dict: Dict[str, List[ForeignKey]] = {}

            for t_name in tqdm(table_names, desc="Downloading tables"):
                sql_table = sa.Table(t_name, remote_md)

                dtypes: Dict[str, str] = {}
                sql_types_dict: Dict[str, sa.types.TypeEngine] = {}

                for c in sql_table.columns:
                    dtype, sql_type = resolve_column_dtype(c)

                    if dtype is not None:
                        dtypes[c.name] = dtype
                        sql_types_dict[c.name] = sql_type
                    else:
                        warnings.warn(
                            f"Unknown data type {c.type} in {t_name}.{c.name}; "
                            "the column is skipped.",
                            stacklevel=2,
                        )

                statement = sa.select(sql_table.columns)
                query = statement.compile(remote_con.engine)
                df = pd.read_sql_query(str(query), con=remote_con, dtype=dtypes)

                time_col = self.time_col_dict.get(t_name, None)
                if time_col is not None and time_col not in df.columns:
                    raise ValueError(
                        f"Configured time column '{time_col}' not found in "
                        f"table '{t_name}'."
                    )

                for col, sql_type in sql_types_dict.items():
                    if sql_type in SQL_DATE_TYPES or time_col == col:
                        try:
                            df[col] = pd.to_datetime(df[col])
                        except (
                            ValueError,
                            TypeError,
                            pd.errors.OutOfBoundsDatetime,
                        ) as e:
                            warnings.warn(
                                f"Could not convert {t_name}.{col} to datetime: {e}",
                                stacklevel=2,
                            )

                    if SQL_DATE_MAP.get(sql_type, None) is not None:
                        try:
                            df[col] = df[col].astype(SQL_DATE_MAP[sql_type], errors="raise")
                        except (
                            ValueError,
                            TypeError,
                            pd.errors.OutOfBoundsDatetime,
                        ) as e:
                            warnings.warn(
                                f"Could not convert {t_name}.{col} to "
                                f"{SQL_DATE_MAP[sql_type]}: {e}",
                                stacklevel=2,
                            )

                # Create index column used as artificial primary key
                df.index.name = "__PK__"
                df.reset_index(inplace=True)

                df_dict[t_name] = df

                fk_dict[t_name] = inspector.get_foreign_keys(t_name)
        finally:
            remote_con.close()
            remote_con.engine.dispose()

        table_dict: Dict[str, Table] = {}

        # Re-index keys as RelBench do not support composite keys.
        # Also this way all original columns are preserved.
        for t_name in table_names:
            fkey_col_to_pkey_table: Dict[str, str] = {}

            for fk in fk_dict[t_name]:
                fk_col, fk_name = reindex_fk(
                    df_dict, t_name, fk.src_columns, fk.ref_table, fk.ref_columns
                )

                fkey_col_to_pkey_table[fk_name] = fk.ref_table
                df_dict[t_name][fk_name] = fk_col

            table_dict[t_name] = Table(
                df=df_dict[t_name],
                fkey_col_to_pkey_table=fkey_col_to_pkey_table,
                pkey_col="__PK__",
                time_col=self.time_col_dict.get(t_name, None),
            )

        db = Database(table_dict)

        # Allow custom modifications here (e.g. dropping columns, etc.)
        db = self.customize_db(db)

        # Remove original primary and foreign keys
        if not self.keep_original_keys:
            for t_name in table_names:
                if t_name not in db.table_dict:
                    continue

                sql_table = sa.Table(t_name, remote_md)
                table = db.table_dict[t_name]
                drop_cols = set()

                # Drop primary key columns
                if (
                    not self.keep_original_compound_keys
                    or len(sql_table.primary_key.columns) == 1
                ):
                    drop_cols |= {c.name for c in sql_table.primary_key.columns}

                for fk in sql_table.foreign_key_constraints:
                    if fk.referred_table.name not in db.table_dict:
                        continue

                    if not self.keep_original_compound_keys or len(fk.columns) == 1:
                        # Drop foreign key columns
                        drop_cols |= {c.name for c in fk.columns}

                # customize_db may have dropped some of these columns already.
                drop_cols &= set(table.df.columns)
                table.df.drop(columns=drop_cols, inplace=True)

        return db
