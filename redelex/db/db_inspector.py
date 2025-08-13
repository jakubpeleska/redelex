from typing import Dict, List, Set, Tuple

from sqlalchemy import Connection, Engine, inspect
from sqlalchemy.types import TypeEngine

from .column_types import ForeignKey

__all__ = ["DBInspector"]


class DBInspector:
    """
    A simplified helper class that allows to retrieve select basic information about a database,
    its tables, columns, and values.
    """

    def __init__(
        self,
        connection: Connection,
    ):
        """
        Initializes a new instance of the DBInspector class.

        Args:
            connection (Connection): The database connection - instance of SQLAlchemy's `Connection` class.
        """
        self._connection = connection
        self._inspect = inspect(self._connection.engine)

    @property
    def connection(self) -> Connection:
        return self._connection

    @property
    def engine(self) -> Engine:
        return self._connection.engine

    def get_tables(self) -> Set[str]:
        """
        Retrieves the names of all tables in the database.

        Returns:
            Set[str]: A set of table names.
        """
        out = self._inspect.get_table_names()
        return set(out)

    def get_columns(self, table: str) -> Dict[str, TypeEngine]:
        """
        Retrieves the columns of a specific table in the database.

        Args:
            table (str): The name of the table.

        Returns:
            Dict[str, TypeEngine]: A dictionary mapping column names to their SQLAlchemy types.
        """
        out = {col["name"]: col["type"] for col in self._inspect.get_columns(table)}
        return out

    def get_table_column_pairs(self) -> Set[Tuple[str, str]]:
        """
        Retrieves all table-column pairs in the database.

        Returns:
            Set[Tuple[str, str]]: A set of tuples representing table-column pairs.
        """
        out = set()

        for tbl in self.get_tables():
            out |= {(tbl, col) for col in self.get_columns(tbl).keys()}

        return out

    def get_primary_key(self, table: str) -> Set[str]:
        """
        Retrieves the primary key columns of a specific table in the database.

        Args:
            table (str): The name of the table.

        Returns:
            Set[str]: A set of primary key column names.
        """
        return set(self._inspect.get_pk_constraint(table)["constrained_columns"])

    def get_foreign_keys(self, table: str) -> List[ForeignKey]:
        """
        Retrieves the foreign key constraints of a specific table in the database.

        Args:
            table (str): The name of the table.

        Returns:
            List[ForeignKey]: A dictionary mapping sets of constrained columns to their corresponding ForeignKey objects.
        """
        return [
            ForeignKey(
                src_columns=fk["constrained_columns"],
                ref_table=fk["referred_table"],
                ref_columns=fk["referred_columns"],
            )
            for fk in self._inspect.get_foreign_keys(table)
        ]
