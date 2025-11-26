from abc import ABC, abstractmethod

import pandas as pd

from relbench.base import Database

from .foreign_key import ForeignKey
from .schema import DBSchema


class DBInterface(ABC):
    @property
    @abstractmethod
    def table_names(self) -> list[str]:
        pass

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def get_primary_key(self, table_name: str) -> list[str]:
        pass

    @abstractmethod
    def get_foreign_keys(self, table_name: str) -> list[ForeignKey]:
        pass

    @abstractmethod
    def get_schema(self) -> DBSchema:
        pass

    @abstractmethod
    def sql(self, query: str) -> "pd.DataFrame":
        pass

    @abstractmethod
    def get_table(self, table_name: str) -> "pd.DataFrame":
        pass

    def get_tables(self) -> dict[str, "pd.DataFrame"]:
        return {name: self.get_table(name) for name in self.table_names()}

    @abstractmethod
    def get_relbench_db(self) -> Database:
        pass
