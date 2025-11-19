from typing import Dict, List, Optional, Union

import json


import pandas as pd
import networkx as nx

import torch
from sentence_transformers import SentenceTransformer

from torch_frame import stype
from torch_frame.config import TextEmbedderConfig

from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx
from torch_geometric.loader import NeighborLoader

from relbench.base import Database, Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset_names, get_dataset
from relbench.tasks import get_task_names, get_task

from redelex.data import make_pkey_fkey_graph, get_node_train_table_input
from redelex.datasets import CTUDataset, DBDataset
from redelex.tasks import CTUBaseEntityTask
from redelex.utils import convert_timedelta, guess_schema


class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> torch.Tensor:
        return self.model.encode(sentences, convert_to_tensor=True)


def max_multiplicity(df: pd.DataFrame, fk_col: str):
    mltp = df[fk_col].dropna().value_counts().max()
    return mltp if pd.isna(mltp) == False else 0  # noqa E712


def db_relation_multiplicity(db: Database):
    multiplicity = {
        tname: {
            "one_to_one": 0,
            "one_to_many": 0,
            "many_to_one": 0,
            "like_many_to_many": False,
            "is_link_table": False,
        }
        for tname in db.table_dict.keys()
    }

    for tname, table in db.table_dict.items():
        for fk, fktname in table.fkey_col_to_pkey_table.items():
            mltp = max_multiplicity(table.df, fk)
            if mltp > 1:
                multiplicity[tname]["many_to_one"] += 1
                multiplicity[fktname]["one_to_many"] += 1
            elif mltp == 1:
                multiplicity[tname]["one_to_one"] += 1
                multiplicity[fktname]["one_to_one"] += 1

    for tname, table in db.table_dict.items():
        if (
            multiplicity[tname]["many_to_one"] >= 2
            and multiplicity[tname]["one_to_many"] == 0
        ):
            multiplicity[tname]["like_many_to_many"] = True
            if (
                multiplicity[tname]["one_to_one"] == 0
                and len(table.df.columns) - 1 - len(table.fkey_col_to_pkey_table) <= 0
            ):
                multiplicity[tname]["is_link_table"] = True

    return multiplicity


def db_stype_multiplicity(dataset: Dataset):
    db = dataset.get_db(upto_test_timestamp=False)
    if isinstance(dataset, DBDataset):
        col_to_stype_dict = guess_schema(db, dataset.get_schema())
    else:
        col_to_stype_dict = guess_schema(db)

    multiplicity = {
        tname: {
            stype.categorical.name: 0,
            stype.numerical.name: 0,
            stype.text_embedded.name: 0,
            stype.timestamp.name: 0,
        }
        for tname in col_to_stype_dict.keys()
    }
    for tname, col_to_stype in col_to_stype_dict.items():
        for s in col_to_stype.values():
            if s in stype:
                multiplicity[tname][s.name] += 1

    return multiplicity


all_datasets = get_dataset_names()
ctu_datasets = list(filter(lambda x: x.startswith("ctu"), all_datasets))
relbench_datasets = list(filter(lambda x: x.startswith("rel"), all_datasets))

info: Dict[str, Dict[str, Union[int, float, str]]] = {}
# info = pd.read_csv("task-info.csv", index_col="task").T.to_dict()
print(info)

cache_dir = ".cache"

for dataset_name in all_datasets:
    for task_name in get_task_names(dataset_name):
        if task_name in info:
            info[task_name]["task"] = task_name
            continue
        task: Union[EntityTask, CTUBaseEntityTask] = get_task(dataset_name, task_name)
        if task.task_type == TaskType.LINK_PREDICTION:
            continue

        dataset = get_dataset(dataset_name)
        try:
            if isinstance(dataset, CTUDataset):
                db = task.get_sanitized_db(upto_test_timestamp=False)
                cache_path = f"{cache_dir}/{dataset_name}/{task_name}"
            else:
                db = dataset.get_db(upto_test_timestamp=False)
                cache_path = f"{cache_dir}/{dataset_name}"
        except Exception as e:
            print(f"Error: {e}")
            continue

        convert_timedelta(db)

        print(f"Processing {dataset_name} - {task_name}")

        info[task_name] = {}
        info[task_name]["dataset"] = dataset_name
        info[task_name]["task"] = task_name

        entity_table = db.table_dict[task.entity_table]

        info[task_name]["entity_fks"] = len(entity_table.fkey_col_to_pkey_table)

        info[task_name]["entity_fact_cols"] = (
            len(entity_table.df.columns) - info[task_name]["entity_fks"] - 1
        )

        rel_mltpl = db_relation_multiplicity(db)
        info[task_name].update(
            {f"target_{k}": v for k, v in rel_mltpl[task.entity_table].items()}
        )

        stype_mltpl = db_relation_multiplicity(db)
        info[task_name].update(
            {f"target_{k}": v for k, v in stype_mltpl[task.entity_table].items()}
        )

        try:
            with open(f"{cache_path}/stypes.json", "r") as f:
                col_to_stype_dict = json.load(f)
            for table, col_to_stype in col_to_stype_dict.items():
                for col, stype_str in col_to_stype.items():
                    if isinstance(stype_str, str):
                        col_to_stype[col] = stype(stype_str)
        except FileNotFoundError:
            schema = {}
            if isinstance(dataset, CTUDataset):
                schema = dataset.get_schema()
            col_to_stype_dict = guess_schema(db, schema)

        data, col_stats_dict = make_pkey_fkey_graph(
            db,
            col_to_stype_dict=col_to_stype_dict,
            text_embedder_cfg=TextEmbedderConfig(
                text_embedder=GloveTextEmbedding(device=torch.device("cpu")),
                batch_size=256,
            ),
            cache_dir=f"{cache_path}/materialized",
        )

        for split in ["train", "val", "test"]:
            table = task.get_table(split, mask_input_cols=False)
            info[task_name][f"n_{split}"] = len(table.df)

            if split != "train":
                continue

            table_input = get_node_train_table_input(table=table, task=task)
            is_temporal = table_input.time is not None
            loader = NeighborLoader(
                data,
                num_neighbors=[max(int(16 / 2**i), 1) for i in range(20)],
                time_attr="time" if is_temporal else None,
                input_nodes=table_input.nodes,
                input_time=table_input.time if is_temporal else None,
                transform=table_input.transform,
                batch_size=1,
                temporal_strategy="uniform",
                shuffle=True,
            )

            N = 50
            i = 0
            entity_eccentricity = []
            sample_density = []
            for b, i in zip(loader, range(N)):
                b: HeteroData = b
                G: nx.MultiDiGraph = to_networkx(b, to_multi=True)
                e = nx.eccentricity(G.reverse(), b.node_offsets[task.entity_table])
                d = nx.density(G)
                entity_eccentricity.append(e)
                sample_density.append(d)

            info[task_name]["entity_eccentricity"] = (
                torch.Tensor(entity_eccentricity).mean().item()
            )
            info[task_name]["sample_density"] = torch.Tensor(sample_density).mean().item()

        df = pd.DataFrame(info).T
        df.to_csv("./task-info.csv", index=False)


df = pd.DataFrame(info).T

df.to_csv("./task-info.csv", index=False)
