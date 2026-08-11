from typing import Any, Dict, Optional

from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

import pandas as pd

import numpy as np

import torch

from torch_geometric.loader import NeighborLoader

from relbench.datasets import get_dataset
from relbench.tasks import get_task

from redelex.data.graph import make_pkey_fkey_graph

from notebooks.utils import get_potato_client, get_experiment_runs_df

from experiments.continuous_learning.utils import (
    get_attribute_schema,
    get_text_embedder,
    get_table_input,
)
from experiments.continuous_learning.models import HeterogeneousSAGE
from experiments.continuous_learning.continuous_task import ContinuousWrapper


def generate_all_predictions_df(
    dataset_name: str,
    task_name: str,
    mlflow_experiment: str,
    cache_dir: str = ".cache",
    batch_size: int = 128,
    num_neighbors: int = 32,
    gnn_channels: int = 128,
    gnn_layers: int = 2,
    gnn_aggr: str = "sum",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mlflow_client = get_potato_client()
    runs_df = get_experiment_runs_df(
        mlflow_client,
        mlflow_experiment,
        filter_string=f"params.dataset_name = '{dataset_name}' and params.task_name = '{task_name}' and attributes.status = 'FINISHED'",
    )

    if runs_df.empty:
        print("No finished runs found matching the criteria.")
        return None

    print(f"Found {len(runs_df)} finished runs.")

    # Setup Data and Task
    cache_path = Path(cache_dir).absolute() / dataset_name
    dataset = get_dataset(dataset_name, download=False)
    db = dataset.get_db(upto_test_timestamp=False)

    task = get_task(dataset_name, task_name)
    wrapped_task = ContinuousWrapper(task)

    text_embedder = get_text_embedder("glove", device=torch.device("cpu"))
    attribute_schema = get_attribute_schema(f"{cache_path}/attribute-schema.json", db)

    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=attribute_schema,
        text_embedder=text_embedder,
        cache_dir=f"{cache_path}/materialized",
    )

    # Create a table covering the entire dataset duration
    full_input = get_table_input(wrapped_task.full_table, task)

    # Create the full dataloader
    full_loader = NeighborLoader(
        data,
        num_neighbors=[int(num_neighbors / 2**i) for i in range(gnn_layers)],
        time_attr="time",
        input_nodes=full_input.nodes,
        input_time=full_input.time,
        transform=full_input.transform,
        batch_size=batch_size,
        temporal_strategy="uniform",
        shuffle=False,
    )

    # Initialize the model architecture
    model = HeterogeneousSAGE(
        data=data,
        col_stats_dict=col_stats_dict,
        gnn_channels=gnn_channels,
        gnn_layers=gnn_layers,
        gnn_aggr=gnn_aggr,
    ).to(device)

    data_dir = Path(f"data/{mlflow_experiment}").absolute()
    data_dir.mkdir(parents=True, exist_ok=True)
    
    wrapped_task.full_table.df.to_csv(
        data_dir / f"{dataset_name}_{task_name}_predictions.csv", index=False
    )
    
    results_df = pd.read_csv(
        data_dir / f"{dataset_name}_{task_name}_predictions.csv"
    )
    
    # Iterate through all runs and perform inference
    for _, run in tqdm(runs_df.iterrows(), total=len(runs_df), desc="Evaluating Runs"):
        run_id = run["_run_id"]
        increment = run.get("increment", "unknown")
        

        # Resolve weights path
        weights_path = None
        if "model_save_dir" in run and pd.notna(run["model_save_dir"]):
            weights_path = Path(run["model_save_dir"]) / "best_model.pt"

        if weights_path is None or not weights_path.exists():
            print(f"\nSkipping run_id {run_id} - Model weights not found.")
            continue
        
        col_name = f"{increment}_{run_id}"
        if col_name in results_df.columns:
            print(f"\nSkipping run_id {run_id} - Predictions already exist in CSV.")
            continue

        # Load weights and set to eval
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.to(device)
        model.eval()

        try:
            results_df = pd.read_csv(
                f"data/{mlflow_experiment}/{dataset_name}_{task_name}_predictions.csv"
            )
        except FileNotFoundError:
            print(
                f"Predictions CSV not found for dataset {dataset_name} and task {task_name}. Creating empty dataframe."
            )
            results_df = pd.DataFrame()

        all_preds = []
        with torch.no_grad():
            for batch in full_loader:
                batch = batch.to(device)

                # Forward pass
                preds = model(batch, task.entity_table)

                all_preds.append(preds.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)

        # Flatten array if the task returns shape (N, 1)
        if all_preds.ndim > 1 and all_preds.shape[1] == 1:
            all_preds = all_preds.flatten()

        # Add predictions as a new column: e.g., pred_inc1_8e4a9b...
        results_df[col_name] = all_preds
        results_df.to_csv(
            f"data/{mlflow_experiment}/{dataset_name}_{task_name}_predictions.csv",
            index=False,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--mlflow_experiment", type=str, default=None)

    args = parser.parse_args()
    print(args)

    generate_all_predictions_df(
        dataset_name=args.dataset,
        task_name=args.task,
        mlflow_experiment=args.mlflow_experiment,
    )
