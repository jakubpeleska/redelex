from typing import Any, Dict, Literal, Optional

import torch
from torch import Tensor

from torch_frame.data.stats import StatType
from torch_frame.nn import ResNet

from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder

from redelex.nn.encoders import LinearRowEncoder, PerFeatureRowEncoder
from redelex.nn.models import DBFormer


class RDLModel(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        tabular_model: Literal["resnet", "linear"] = "resnet",
        tabular_channels: int = 128,
        tabular_kwargs: Optional[Dict[str, Any]] = None,
        rgnn_model: Literal["sage", "dbformer"] = "sage",
        rgnn_channels: int = 128,
        rgnn_layers: int = 2,
        rgnn_aggr: Optional[str] = "sum",
        rgnn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        def get_tabular_model():
            if tabular_model == "resnet":
                if rgnn_model == "dbformer":
                    raise ValueError(
                        "ResNet is not compatible with DBFormer. Use linear instead."
                    )
                return ResNet, {
                    "channels": tabular_channels,
                    "num_layers": 4,
                }
            elif tabular_model == "linear":
                if rgnn_model == "dbformer":
                    return PerFeatureRowEncoder, {"channels": tabular_channels}

                return LinearRowEncoder, {"channels": tabular_channels}
            else:
                raise ValueError(f"Unknown tabular_model: {tabular_model}")

        tabular_cls, _tabular_kwargs = get_tabular_model()
        _tabular_kwargs.update(tabular_kwargs or {})

        self.node_types = data.node_types
        self.edge_types = data.edge_types

        self.encoder = HeteroEncoder(
            channels=rgnn_channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
            torch_frame_model_cls=tabular_cls,
            torch_frame_model_kwargs=_tabular_kwargs,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=rgnn_channels,
        )

        _rgnn_kwargs = dict(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=rgnn_channels,
            aggr=rgnn_aggr,
            num_layers=rgnn_layers,
            **(rgnn_kwargs or {}),
        )

        def get_rgnn_model(model: str):
            if model == "sage":
                return HeteroGraphSAGE(**_rgnn_kwargs)
            elif model == "dbformer":
                return DBFormer(
                    **_rgnn_kwargs,
                    col_stats_dict=col_stats_dict,
                    with_output_transform=True,
                )
            else:
                raise ValueError(f"Unknown rgnn_model: {model}")

        self.rgnn_model = rgnn_model

        self.rgnn = get_rgnn_model(rgnn_model)

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.rgnn.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: Optional[NodeType] = None,
        tf_attr: Optional[str] = "tf",
    ) -> Dict[NodeType, Tensor]:
        x_dict = self.encoder(batch.collect(tf_attr))

        if entity_table and hasattr(batch[entity_table], "seed_time"):
            seed_time = batch[entity_table].seed_time
            rel_time_dict = self.temporal_encoder(
                seed_time, batch.collect("time"), batch.collect("batch")
            )

            if self.rgnn_model == "dbformer":
                for node_type, rel_time in rel_time_dict.items():
                    x_dict[node_type] = x_dict[node_type] + rel_time[:, None, :]
            else:
                for node_type, rel_time in rel_time_dict.items():
                    x_dict[node_type] = x_dict[node_type] + rel_time

        x_dict = self.rgnn(
            x_dict,
            batch.edge_index_dict,
            # batch.num_sampled_nodes_dict,
            # batch.num_sampled_edges_dict,
        )

        return x_dict
