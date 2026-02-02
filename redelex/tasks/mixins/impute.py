from typing import Callable

import pandas as pd

from relbench.base import Database, Table, TaskType

from .db_modify import ModifyDBTaskMixin
from .entity import EntityTaskMixin


class ImputeEntityTaskMixin(ModifyDBTaskMixin, EntityTaskMixin):
    r"""Mixin class for allowing to modify underlying database for a task.
    Attributes:
        removed_entity_cols: list of entity columns to be removed from the
            entity table.
        Other attributes are inherited from ModifyDBTaskMixin and EntityTaskMixin.
    """

    removed_entity_cols: list[str] = []
    entity_col: str
    entity_table: str
    target_col: str
    task_type: TaskType

    _target_mapping: Callable[[pd.Series], pd.Series] = None
    _target_dtype: type = None

    def _init_target_mapping(self, df: pd.DataFrame) -> None:
        if self.task_type in [
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        ]:
            _, target_values = df[self.target_col].factorize(
                sort=True, use_na_sentinel=True
            )

            def target_map(x):
                if pd.isna(x):
                    return -1
                else:
                    return target_values.get_loc(x)

            self._target_mapping = target_map
        else:
            self._target_mapping = lambda x: x

        if self.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION]:
            self._target_dtype = float
        elif self.task_type in [TaskType.MULTICLASS_CLASSIFICATION]:
            self._target_dtype = int

    def _make_modified_db(self, db: Database) -> Database:
        r"""
        Modify the database for the task.
        Args:
            db: The database to make modifications on.
        Returns:
            A modified database.
        """

        remove_cols = list(set([self.target_col, *self.removed_entity_cols]))

        db.table_dict[self.entity_table].df.drop(columns=remove_cols, inplace=True)

        return db

    def filter_dangling_entities(self, table: Table) -> Table:
        return table


__all__ = ["ImputeEntityTaskMixin"]
