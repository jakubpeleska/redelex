import json
import os
from pathlib import Path
from typing import Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from relbench.base import Database, Table

__all__ = ["save_table", "load_table", "save_db", "load_db"]

# relbench 3 dropped `Table.save/load` and `Database.save/load`; these are ports of
# the relbench 2.1 methods, so the on-disk cache format stays the same.


def save_table(table: Table, path: Union[str, os.PathLike]) -> None:
    r"""Save the table to a parquet file.

    Stores other attributes as parquet metadata.
    """
    assert str(path).endswith(".parquet")
    metadata = {
        "fkey_col_to_pkey_table": table.fkey_col_to_pkey_table,
        "pkey_col": table.pkey_col,
        "time_col": table.time_col,
    }

    # Convert DataFrame to a PyArrow Table
    pa_table = pa.Table.from_pandas(table.df, preserve_index=False)

    # Add metadata to the PyArrow Table
    metadata_bytes = {
        key: json.dumps(value).encode("utf-8") for key, value in metadata.items()
    }

    pa_table = pa_table.replace_schema_metadata(
        {**pa_table.schema.metadata, **metadata_bytes}
    )

    # Write the PyArrow Table to a Parquet file using pyarrow.parquet
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa_table, path)


def load_table(path: Union[str, os.PathLike]) -> Table:
    r"""Load a table from a parquet file."""
    assert str(path).endswith(".parquet")

    # Read the Parquet file using pyarrow
    pa_table = pq.read_table(path)
    df = pa_table.to_pandas()
    # DuckDB currently cannot ingest pandas StringDtype columns reliably.
    # Convert them to plain object dtype on load.
    for col in df.columns:
        if isinstance(df[col].dtype, pd.StringDtype):
            df[col] = df[col].astype(object)

    # Extract metadata
    metadata_bytes = pa_table.schema.metadata
    metadata = {
        key.decode("utf-8"): json.loads(value.decode("utf-8"))
        for key, value in metadata_bytes.items()
        if key in [b"fkey_col_to_pkey_table", b"pkey_col", b"time_col"]
    }
    return Table(
        df=df,
        fkey_col_to_pkey_table=metadata["fkey_col_to_pkey_table"],
        pkey_col=metadata["pkey_col"],
        time_col=metadata["time_col"],
    )


def save_db(db: Database, path: Union[str, os.PathLike]) -> None:
    r"""Save the database to a directory.

    Simply saves each table individually with the table name as base name of file.
    """

    for name, table in db.table_dict.items():
        save_table(table, f"{path}/{name}.parquet")


def load_db(path: Union[str, os.PathLike]) -> Database:
    r"""Load a database from a directory of tables in parquet files."""

    table_dict = {}
    for table_path in Path(path).glob("*.parquet"):
        table = load_table(table_path)
        table_dict[table_path.stem] = table

    return Database(table_dict)
