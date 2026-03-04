import torch

from relbench.tasks import get_task

from redelex.tasks import mixins

from experiments.universal_encoder.universal_encoder_pretrained import (
    get_dataset_data,
    get_task_data,
    PRETRAIN_TASKS,
)
from experiments.universal_encoder.utils import get_text_embedder

cache_dir = ".cache"
text_embedder = get_text_embedder("glove", torch.device("cpu"))

loader_dicts = {}

for dataset_name, task_names in PRETRAIN_TASKS.items():
    cache_path = f"{cache_dir}/{dataset_name}"
    target = None
    if len(task_names) == 1:
        task = get_task(dataset_name, task_names[0])
        if isinstance(task, mixins.ImputeEntityTaskMixin):
            target = (task.entity_table, task.target_col)

    data, col_stats_dict, tensor_stats_dict, name_embeddings_dict = get_dataset_data(
        dataset_name,
        cache_path,
        text_embedder,
        torch.device("cpu"),
        target=target,
    )

    for task_name in task_names:
        task = get_task(dataset_name, task_name)
        name = f"{dataset_name}-{task_name}"
        task, loader_dict = get_task_data(
            task,
            name,
            data,
            name_embeddings_dict,
            tensor_stats_dict,
            col_stats_dict,
            {
                "gnn_layers": 2,
                "num_neighbors": 16,
                "batch_size": 128,
            },
        )
        loader_dicts[name] = loader_dict
