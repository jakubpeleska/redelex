from collections import defaultdict
from functools import lru_cache
from typing import List

from redelex.datasets.registry import get_cache_dir, get_dataset

from .mixins import BaseTask

# relbench 3 dropped its task registry; this is a port of the relbench 2.1 one,
# without the download from the RelBench server.

__all__ = ["task_registry", "register_task", "get_task_names", "get_task"]

task_registry = defaultdict(dict)


def register_task(
    dataset_name: str,
    task_name: str,
    cls: BaseTask,
    *args,
    **kwargs,
) -> None:
    r"""Register an instantiation of a :class:`BaseTask` subclass with the given name.

    Args:
        dataset_name: The name of the dataset.
        task_name: The name of the task.
        cls: The class of the task.
        args: The arguments to instantiate the task.
        kwargs: The keyword arguments to instantiate the task.

    The name is used to enable caching. `cache_dir` is added to kwargs by default.
    If you want to override it, you can pass `cache_dir` as a keyword argument in
    `kwargs`.
    """

    cache_dir = f"{get_cache_dir()}/{dataset_name}/tasks/{task_name}"
    kwargs = {"cache_dir": cache_dir, **kwargs}
    task_registry[dataset_name][task_name] = (cls, args, kwargs)


def get_task_names(dataset_name: str) -> List[str]:
    r"""Return a list of names of the registered tasks for the given dataset."""
    return list(task_registry[dataset_name].keys())


@lru_cache(maxsize=None)
def get_task(dataset_name: str, task_name: str) -> BaseTask:
    r"""Return a task object by name.

    Args:
        dataset_name: The name of the dataset.
        task_name: The name of the task.

    Returns:
        BaseTask: The task object.

    The task tables are computed from the database on the first ``get_table()``
    call and cached in the task's cache directory; later calls use the cache.
    """

    dataset = get_dataset(dataset_name)
    cls, args, kwargs = task_registry[dataset_name][task_name]
    return cls(dataset, *args, **kwargs)
