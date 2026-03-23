from mlflow.tracking import MlflowClient
from mlflow.entities import Run

import pandas as pd


def get_potato_client():
    return MlflowClient(tracking_uri="http://potato.felk.cvut.cz:2222")


def get_experiment_runs(client: MlflowClient, experiment_name: str) -> list[Run]:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    experiment_id = experiment.experiment_id

    next_token = -1
    all_runs = []
    while next_token is not None:
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="status != 'FAILED'",
            max_results=1000,
            page_token=next_token if next_token != -1 else None,
        )
        next_token = runs.token
        all_runs.extend(runs.to_list())
    return all_runs


def get_experiment_runs_df(client: MlflowClient, experiment_name: str) -> pd.DataFrame:
    all_runs = get_experiment_runs(client, experiment_name)
    runs_dict = []
    for r in all_runs:
        r_info = r.info.__dict__
        r_data = {k: v for d in r.data.to_dictionary().values() for k, v in d.items()}
        runs_dict.append({**r_info, **r_data})

    df = pd.DataFrame(runs_dict)
    return df


def get_run_metrics(client: MlflowClient, run_id: str, metrics: list[str]) -> pd.DataFrame:
    metrics_dict = {}
    for metric in metrics:
        metric_history = client.get_metric_history(run_id, metric)
        for m in metric_history:
            if m.step not in metrics_dict:
                metrics_dict[m.step] = {}
            metrics_dict[m.step][m.key] = m.value

    df: pd.DataFrame = pd.DataFrame.from_dict(metrics_dict, orient="index")
    df.reset_index(inplace=True)
    df.rename(columns={"index": "step"}, inplace=True)
    df["run_id"] = run_id
    df = df[["run_id", "step"] + metrics]
    return df
