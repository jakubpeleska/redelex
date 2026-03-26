from copy import deepcopy
from typing import Optional

from relbench.base import Database

from .base import BaseTask


class ModifyDBTaskMixin(BaseTask):
    r"""Mixin class for allowing to modify underlying database for a task.

    Attributes are inherited from BaseTask.
    """

    def _make_modified_db(self, db: Database) -> Database:
        r"""
        Modify the database for the task.
        Args:
            db: The database to make modifications on.
        Returns:
            A modified database.

        To be implemented by subclass.
        """

        raise NotImplementedError

    def make_modified_db(
        self, db: Optional[Database] = None, inplace: bool = False
    ) -> Database:
        r"""
        Make a modified database using the task definition.
        Args:
            db: The database to make modifications on. If None, use the database
                from the dataset.
            inplace: If True, modify the database in place. Otherwise, return a new
                modified copy of the database.
        """

        if db is None:
            db = self.dataset.get_db(upto_test_timestamp=False)

        modified_db = db if inplace else deepcopy(db)

        return self._make_modified_db(modified_db)


__all__ = ["ModifyDBTaskMixin"]
