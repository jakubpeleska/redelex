import duckdb
import pandas as pd

from relbench.base import Database, Dataset as RelbenchDataset

from .foreign_key import ForeignKey
from .interface import DBInterface
from .schema import DBSchema, TableSchema


class RelbenchDBInterface(DBInterface):
    def __init__(self, relbench_dataset: RelbenchDataset):
        self.relbench_dataset = relbench_dataset
        self.db = None

        super().__init__()

    def connect(self):
        self.db = self.relbench_dataset.get_db(False)
        for tname, table in self.db.table_dict.items():
            duckdb.register(tname, table.df)

    def table_names(self) -> list[str]:
        return list(self.db.table_dict.keys())

    def get_primary_key(self, table_name: str) -> list[str]:
        return [self.db.table_dict[table_name].pkey_col]

    def get_foreign_keys(self, table_name: str) -> list[ForeignKey]:
        return [
            ForeignKey([fk], ref_table, [self.db.table_dict[table_name].pkey_col])
            for fk, ref_table in self.db.table_dict[
                table_name
            ].fkey_col_to_pkey_table.items()
        ]

    def get_schema(self) -> DBSchema:
        table_schemas = {}
        for tname, table in self.db.table_dict.items():
            pk = [table.pkey_col]
            fks = self.get_foreign_keys(tname)
            type_dict = {
                col: str(dtype) for col, dtype in zip(table.df.columns, table.df.dtypes)
            }
            table_schemas[tname] = TableSchema(
                name=tname, pk=pk, fks=fks, type_dict=type_dict
            )
        return DBSchema(table_schemas=table_schemas)

    def sql(self, query: str) -> "pd.DataFrame":
        if self.db is None:
            self.connect()
        return duckdb.sql(query).df()

    def get_table(self, table_name: str) -> "pd.DataFrame":
        return self.db.table_dict[table_name].df

    def get_relbench_db(self) -> Database:
        return self.db

    def close(self):
        for tname in self.db.table_dict.keys():
            duckdb.unregister(tname)
        self.db = None
