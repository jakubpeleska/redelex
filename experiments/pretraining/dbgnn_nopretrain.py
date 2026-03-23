from typing import Optional

import os
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["RAY_memory_monitor_refresh_ms"] = "0"

from argparse import ArgumentParser

import ray
from ray import tune, train as ray_train

import numpy as np

import torch


from relbench.base import TaskType
from relbench.datasets import get_dataset
from relbench.tasks import get_task, get_task_names

from redelex.data import make_pkey_fkey_graph

from experiments.pretraining.dbgnn_pretrain import run_task_experiment
from experiments.pretraining.utils import get_attribute_schema, get_text_embedder


def run_ray_tuner(
    dataset_name: str,
    rgnn_model: str,
    tabular_model: str,
    ray_address: Optional[str] = None,
    ray_storage_path: Optional[str] = None,
    ray_experiment_name: Optional[str] = None,
    mlflow_uri: Optional[str] = None,
    mlflow_experiment: str = "pelesjak_test_experiment",
    num_samples: Optional[int] = 1,
    num_gpus: int = 0,
    num_cpus: int = 1,
    random_seed: int = 42,
    cache_dir: str = ".cache",
):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if num_gpus > 0 and ray_address == "local":
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

        nvmlInit()
        free_memory = [
            int(nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(i)).free)
            for i in range(torch.cuda.device_count())
        ]
        device_idx = np.argsort(free_memory)[::-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_idx[:num_gpus].astype(str))
        print("Free memory:", free_memory, os.environ["CUDA_VISIBLE_DEVICES"])

    ray.init(
        address=ray_address,
        ignore_reinit_error=True,
        log_to_driver=False,
        include_dashboard=False,
        num_cpus=num_cpus if ray_address == "local" else None,
        num_gpus=num_gpus if ray_address == "local" else None,
        # _temp_dir=os.path.join(os.getcwd(), ".tmp")
        # if len(os.path.join(os.getcwd(), ".tmp")) < 107
        # else None,
    )

    cache_path = f"{cache_dir}/{dataset_name}"

    dataset = get_dataset(dataset_name)

    db = dataset.get_db(upto_test_timestamp=False)

    train_db = db.upto(dataset.val_timestamp)
    dataset.validate_and_correct_db(train_db)

    schema_cache_path = f"{cache_path}/attribute-schema.json"
    attribute_schema = get_attribute_schema(schema_cache_path, db)

    materialized_cache_dir = f"{cache_path}/materialized"
    full_data, col_stats_dict = make_pkey_fkey_graph(
        db,
        attribute_schema,
        text_embedder=get_text_embedder("glove"),
        cache_dir=materialized_cache_dir + "/full",
    )
    train_data, _ = make_pkey_fkey_graph(
        train_db,
        attribute_schema,
        text_embedder=get_text_embedder("glove"),
        cache_dir=materialized_cache_dir + "/train",
    )

    # resources = ray.available_resources()

    gpus_used = 0
    cpus_used = 1
    # if "GPU" in resources:
    #     batch_model_size = 4e9
    #     gpu_memory = max(
    #         [
    #             torch.cuda.get_device_properties(i).total_memory
    #             for i in range(torch.cuda.device_count())
    #         ]
    #     )
    #     gpus_used = batch_model_size / gpu_memory

    task_names = get_task_names(dataset_name)
    task_names = [
        task_name
        for task_name in task_names
        if get_task(dataset_name, task_name).task_type != TaskType.LINK_PREDICTION
    ]

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                run_task_experiment,
                full_data=full_data,
                col_stats_dict=col_stats_dict,
                with_pretrained=False,
            ),
            resources={"CPU": cpus_used, "GPU": gpus_used},
        ),
        run_config=ray_train.RunConfig(
            name=ray_experiment_name,
            storage_path=ray_storage_path,
            stop={"time_total_s": 3600 * 24},
            log_to_file=True,
        ),
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            trial_name_creator=lambda trial: f"{dataset_name}_{trial.trial_id}",
            trial_dirname_creator=lambda trial: trial.trial_id,
            max_concurrent_trials=num_cpus,
        ),
        param_space={
            "dataset_name": dataset_name,
            "task_name": tune.grid_search(task_names),
            "mlflow_experiment": mlflow_experiment,
            "mlflow_uri": mlflow_uri,
            "seed": tune.randint(0, 1000),
            # training config
            "max_training_steps": 4000,
            "lr": 0.005 if dataset_name != "rel-trial" else 0.0001,
            "finetune_backbone": tune.grid_search([False, True]),
            # sampling config
            "batch_size": 512,
            "num_neighbors": 128,
            # model config
            "channels": 128,
            "tabular_model": tabular_model,  # tune.grid_search(["resnet", "linear"]),
            "rgnn_model": rgnn_model,
            "rgnn_layers": tune.grid_search([2, 3]),
            "rgnn_aggr": tune.grid_search(["mean", "sum"]),
            # head config
            "head_layers": 2,
            "head_channels": 128,
            "head_norm": "batch_norm",
            "head_dropout": 0.0,
        },
    )
    tuner.fit()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--rgnn_model", choices=["sage", "dbformer"], default="sage")
    parser.add_argument("--tabular_model", choices=["resnet", "linear"], default=None)
    parser.add_argument("--ray_address", type=str, default="local")
    parser.add_argument("--ray_storage", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--mlflow_uri", type=str, default=None)
    parser.add_argument("--mlflow_experiment", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument("--num_cpus", type=int, default=1)

    args = parser.parse_args()
    print(args)
    dataset_name = args.dataset

    run_ray_tuner(
        dataset_name,
        rgnn_model=args.rgnn_model,
        tabular_model=args.tabular_model,
        ray_address=args.ray_address,
        ray_storage_path=(
            os.path.realpath(args.ray_storage)
            if args.ray_storage is not None
            else os.path.realpath(".results")
        ),
        ray_experiment_name=args.run_name,
        mlflow_uri=args.mlflow_uri,
        mlflow_experiment=args.mlflow_experiment,
        random_seed=args.seed,
        num_samples=args.num_samples,
        num_gpus=args.num_gpus,
        num_cpus=args.num_cpus,
    )
