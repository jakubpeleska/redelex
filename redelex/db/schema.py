from attrs import field, define

from .foreign_key import ForeignKey


@define
class DBSchema:
    """
    Represents the schema of a database.
    """

    table_schemas: dict[str, "TableSchema"] = field(converter=dict)
    """
    Schemas of individual tables, keyed by table name
    """

    @property
    def table_names(self) -> list[str]:
        """
        The list of table names in the database
        """
        return list(self.table_schemas.keys())

    def __str__(self):
        out = ""
        for i, tschema in enumerate(self.table_schemas.values()):
            out += f"{i + 1}/{len(self.table_schemas)} {tschema}\n"
        return out


@define
class TableSchema:
    """
    Represents the schema of one table.
    """

    name: str = field()
    """
    The name of the table
    """

    pk: list[str] = field(converter=list)
    """
    The primary key columns
    """

    fks: list[ForeignKey] = field(converter=list)
    """
    The foreign keys
    """

    type_dict: dict[str, str] = field(converter=dict)
    """
    A mapping from column names to their data types
    """

    @property
    def columns(self) -> list[str]:
        """
        The list of column names
        """
        return list(self.type_dict.keys())

    def __str__(self):
        out = f"Table: {self.name}\n"
        out += f"    Primary Key: {self.pk}\n"
        out += "    Foreign Keys:\n"
        for fk in self.fks:
            out += f"    - {fk}\n"
        out += "    Columns:\n"
        for col, dtype in self.type_dict.items():
            out += f"    - {col}: {dtype}\n"
        return out
