from typing import Dict, Optional, Any, Union

import json
from pathlib import Path

import torch

from torch_frame import stype

from relbench.base import Database

from redelex.data import guess_schema, TextEmbedder, GloveTextEmbedder, PotionTextEmbedder
from redelex.db import DBSchema
from redelex.tasks import mixins


def get_text_embedder(
    embedder_name: str, device: Optional[torch.device] = None
) -> TextEmbedder:
    if embedder_name == "glove":
        return GloveTextEmbedder(device=device)
    elif embedder_name == "potion":
        return PotionTextEmbedder(device=device)
    else:
        raise ValueError(f"Text embedder {embedder_name} is not supported")


def get_hyperparams_logging(
    config: dict[str, Any],
) -> dict[str, Union[str, int, float, bool]]:
    hyperparams_logging = {}
    for key, value in config.items():
        if type(value) in [str, int, float, bool]:
            hyperparams_logging[key] = value
    return hyperparams_logging


def get_attribute_schema(
    schema_cache_path: str,
    db: Database,
    db_schema: Optional[DBSchema] = None,
    task: Optional[mixins.BaseTask] = None,
) -> Dict[str, Dict[str, stype]]:
    try:
        with open(schema_cache_path, "r") as f:
            attribute_schema = json.load(f)
        for tname, table_attribute_schema in attribute_schema.items():
            for col, stype_str in table_attribute_schema.items():
                if isinstance(stype_str, str):
                    table_attribute_schema[col] = stype(stype_str)
    except FileNotFoundError:
        if db_schema is not None:
            attribute_schema = guess_schema(db, db_schema, task=task)
        else:
            attribute_schema = guess_schema(db, task=task)
        Path(schema_cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(schema_cache_path, "w") as f:
            json.dump(attribute_schema, f, indent=2, default=str)

    return attribute_schema
