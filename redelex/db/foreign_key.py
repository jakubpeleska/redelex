from attrs import define, field


@define
class ForeignKey:
    """
    Represents one foreign key.
    """

    src_columns: list[str] = field(converter=list)
    """
    The referencing columns (in this table)
    """

    ref_table: str
    """
    The referenced table name
    """

    ref_columns: list[str] = field(converter=list)
    """
    The referenced columns (in the referenced table)
    """
