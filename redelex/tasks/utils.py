from relbench.base import BaseTask as RelBenchTask

from redelex.tasks.mixins import TemporalTaskMixin


def is_temporal_task(task: object) -> bool:
    """Check if the given task is a temporal task."""
    return isinstance(task, TemporalTaskMixin) or isinstance(task, RelBenchTask)
