import os
from functools import lru_cache
from typing import List

from relbench.base import Dataset

# relbench 3 dropped its dataset registry; this is a port of the relbench 2.1 one,
# without the download from the RelBench server.

__all__ = [
    "dataset_registry",
    "get_cache_dir",
    "register_dataset",
    "get_dataset_names",
    "get_dataset",
]

dataset_registry = {}


def get_cache_dir() -> str:
    r"""Return the root cache directory of ReDeLEx.

    Defaults to ``~/.cache/redelex`` and can be overridden with the
    ``REDELEX_CACHE_DIR`` environment variable.
    """
    return os.getenv("REDELEX_CACHE_DIR") or os.path.expanduser("~/.cache/redelex")


def register_dataset(
    name: str,
    cls: Dataset,
    *args,
    **kwargs,
) -> None:
    r"""Register an instantiation of a :class:`Dataset` subclass with the given name.

    Args:
        name: The name of the dataset.
        cls: The class of the dataset.
        args: The arguments to instantiate the dataset.
        kwargs: The keyword arguments to instantiate the dataset.

    The name is used to enable caching. `cache_dir` is added to kwargs by default.
    If you want to override it, you can pass `cache_dir` as a keyword argument in
    `kwargs`.
    """

    cache_dir = f"{get_cache_dir()}/{name}"
    kwargs = {"cache_dir": cache_dir, **kwargs}
    dataset_registry[name] = (cls, args, kwargs)


def get_dataset_names() -> List[str]:
    r"""Return a list of names of the registered datasets."""
    return list(dataset_registry.keys())


@lru_cache(maxsize=None)
def get_dataset(name: str) -> Dataset:
    r"""Return a dataset object by name.

    Args:
        name: The name of the dataset.

    Returns:
        Dataset: The dataset object.

    The database is downloaded from its source on the first ``get_db()`` call and
    cached in the dataset's cache directory; later calls use the cache.
    """

    cls, args, kwargs = dataset_registry[name]
    return cls(*args, **kwargs)
