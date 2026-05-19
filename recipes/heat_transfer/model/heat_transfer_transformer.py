#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import torch
import torch.nn as nn
from pydantic import computed_field

from noether.core.models import Model
from noether.data.schemas import ModelDataSpecs
from noether.modeling.models.transformer import Transformer, TransformerConfig
from noether.modeling.modules.layers import ContinuousSincosEmbed, LinearProjection
from noether.modeling.modules.layers.continuous_sincos_embed import ContinuousSincosEmbeddingConfig
from noether.modeling.modules.layers.linear_projection import LinearProjectionConfig
from noether.modeling.modules.mlp import MLP
from noether.modeling.modules.mlp.mlp import MLPConfig


class HeatTransferTransformerConfig(TransformerConfig):
    """Transformer config for volume-only heat transfer with simulation parameter conditioning."""

    data_specs: ModelDataSpecs

    @computed_field
    def pos_embed_config(self) -> ContinuousSincosEmbeddingConfig:
        return ContinuousSincosEmbeddingConfig(
            hidden_dim=self.hidden_dim,
            input_dim=self.data_specs.position_dim,
        )

    @computed_field
    def conditioning_projection_config(self) -> MLPConfig:
        return MLPConfig(
            input_dim=self.data_specs.conditioning_dims["simulation_parameters"],
            output_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            init_weights="truncnormal002",
        )

    @computed_field
    def out_projection_config(self) -> LinearProjectionConfig:
        return LinearProjectionConfig(
            input_dim=self.hidden_dim,
            output_dim=self.data_specs.total_output_dim,
            init_weights="truncnormal002",
        )


class HeatTransferTransformer(Model):
    """Transformer wrapper for volume-only heat transfer.

    Takes volume anchor positions and simulation parameters as input,
    predicts velocity, temperature, and pressure fields.
    """

    def __init__(self, model_config: HeatTransferTransformerConfig, **kwargs):
        super().__init__(model_config=model_config, **kwargs)

        hidden_dim = model_config.hidden_dim

        self.data_specs = model_config.data_specs

        self.pos_embed = ContinuousSincosEmbed(config=model_config.pos_embed_config)

        self.volume_bias = MLP(config=MLPConfig(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim))

        self.project_simulation_parameters = MLP(config=model_config.conditioning_projection_config)

        self.placeholder = nn.Parameter(torch.rand(1, 1, hidden_dim) / hidden_dim)

        self.backbone = Transformer(config=model_config)

        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.out = LinearProjection(config=model_config.out_projection_config)

    def forward(
        self,
        volume_anchor_position: torch.Tensor,
        simulation_parameters: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = self.pos_embed(volume_anchor_position)
        x = self.volume_bias(x)

        # Broadcast conditioning to every token
        cond = self.project_simulation_parameters(simulation_parameters)
        if cond.ndim == 2:
            cond = cond.unsqueeze(1)
        x = x + cond

        x = x + self.placeholder

        x = self.backbone(x=x, attn_kwargs={})
        x = self.out(self.norm(x))

        # Split output into per-field tensors
        volume_spec = self.data_specs.domains["volume"]
        result: dict[str, torch.Tensor] = {}
        offset = 0
        for field_name, dim in volume_spec.output_dims.items():
            result[f"volume_{field_name}"] = x[..., offset : offset + dim]
            offset += dim

        return result
