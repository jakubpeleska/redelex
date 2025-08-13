from typing import Callable, Dict, List, Optional
from numpy.typing import NDArray

import pandas as pd

from relbench.base import BaseTask, Database, Table


class CTULinkTask(BaseTask):
    timedelta = pd.Timedelta(-1)
    entity_col = "__PK__"
    entity_table: str
    target_col: str

    def make_table(self, db: Database, split: str) -> Table:
        raise NotImplementedError

    def get_sanitized_db(self, upto_test_timestamp: bool = True) -> Database:
        raise NotImplementedError

    def evaluate(
        self,
        pred: NDArray,
        target_table: Optional[Table] = None,
        metrics: Optional[List[Callable[[NDArray, NDArray], float]]] = None,
    ) -> Dict[str, float]:
        raise NotImplementedError

    def filter_dangling_entities(self, table: Table) -> Table:
        raise NotImplementedError
