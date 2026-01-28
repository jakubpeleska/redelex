import sys

sys.path.append("../")

from typing import Dict, Union

import pandas as pd
import networkx as nx
import sqlalchemy as sa

from torch_frame import stype

from relbench.base import Database, Dataset
from relbench.datasets import get_dataset_names, get_dataset

from redelex.data import guess_schema
from redelex.datasets import CTUDataset, DBDataset


def get_info(dataset: CTUDataset):
    url = dataset.get_url(
        "mariadb",
        "pymysql",
        "guest",
        "ctu-relational",
        "relational.fel.cvut.cz",
        3306,
        "meta",
    )
    with dataset.create_remote_connection(url) as conn:
        metadata = sa.MetaData()
        metadata.reflect(bind=conn.engine)

        information = metadata.tables.get("information")

        q = sa.select(
            information.c.domain,
            information.c.is_artificial,
            information.c.database_size,
        ).where(information.c.database_name == dataset.database)

        info = conn.execute(q).fetchone()

        return info._asdict()


def get_db_schema_graph(db: Database):
    G = nx.MultiGraph()
    for tname, table in db.table_dict.items():
        G.add_node(tname)
        for fk, fktname in table.fkey_col_to_pkey_table.items():
            G.add_edge(tname, fktname, name=fk)
    return G


def has_cycle(G: nx.MultiGraph) -> bool:
    try:
        nx.find_cycle(G)
    except nx.exception.NetworkXNoCycle:
        return False
    return True


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
# info = pd.read_csv("ctu_dataset_info.csv").T.to_dict()

for dataset_name in all_datasets:
    print(f"Processing {dataset_name}")
    dataset = get_dataset(dataset_name)
    try:
        db = dataset.get_db(upto_test_timestamp=False)
    except Exception as e:
        print(f"Error: {e}")
        continue

    info[dataset_name] = {}
    info[dataset_name]["dataset"] = dataset_name

    if isinstance(dataset, CTUDataset):
        meta = get_info(dataset)
        meta["db_size_MB"] = meta["database_size"]
        meta.pop("database_size")
        info[dataset_name] = {**info[dataset_name], **meta}

    info[dataset_name]["n_tables"] = len(db.table_dict)
    info[dataset_name]["n_fks"] = len(
        [fk for t in db.table_dict.values() for fk in t.fkey_col_to_pkey_table.keys()]
    )
    info[dataset_name]["n_factual_cols"] = (
        sum([len(t.df.columns) for t in db.table_dict.values()])
        - info[dataset_name]["n_fks"]
        - info[dataset_name]["n_tables"]
    )

    info[dataset_name]["total_n_tuples"] = sum(len(t.df) for t in db.table_dict.values())
    info[dataset_name]["total_n_fk_edges"] = sum(
        t.df[fk].notna().sum()
        for t in db.table_dict.values()
        for fk in t.fkey_col_to_pkey_table.keys()
    )

    G = get_db_schema_graph(db)
    info[dataset_name]["schema_diameter"] = nx.diameter(G)
    info[dataset_name]["has_loops"] = has_cycle(G)

    rel_mltpl = db_relation_multiplicity(db)
    rel_mltpl_df = pd.DataFrame(rel_mltpl).T
    info[dataset_name].update(rel_mltpl_df.sum().to_dict())

    stype_mltpl = db_relation_multiplicity(db)
    stype_mltpl_df = pd.DataFrame(stype_mltpl).T
    info[dataset_name].update(stype_mltpl_df.sum().to_dict())

    df = pd.DataFrame(info).T
    df.to_csv("./dataset-info.csv", index=False)


df = pd.DataFrame(info).T

df.to_csv("./dataset-info.csv", index=False)
