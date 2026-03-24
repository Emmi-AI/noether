#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import torch
import torch.nn as nn

from noether.core.models import Model
from noether.core.schemas.dataset import AeroDataSpecs
from noether.core.schemas.models import TransformerConfig
from noether.core.schemas.modules.layers import (
    ContinuousSincosEmbeddingConfig,
    LinearProjectionConfig,
    RopeFrequencyConfig,
)
from noether.core.schemas.modules.mlp import MLPConfig
from noether.modeling.models import Transformer
from noether.modeling.modules.layers import ContinuousSincosEmbed, LinearProjection, RopeFrequency
from noether.modeling.modules.mlp import MLP


class TransformerWrapperConfig(TransformerConfig):
    """Wrapper config that extends TransformerConfig with data specifications."""

    data_specs: AeroDataSpecs


class TransformerWrapper(Model):
    """Factory-compatible wrapper for the Transformer backbone.

    Handles the end-to-end forward pass for aero CFD tasks: concatenates surface/volume positions, applies positional
    encoding, optional RoPE, optional physics feature injection, surface/volume bias, runs the Transformer backbone,
    projects output, and splits the result back into per-field dictionaries.
    """

    def __init__(self, model_config: TransformerWrapperConfig, **kwargs):
        super().__init__(model_config=model_config, **kwargs)

        hidden_dim = model_config.hidden_dim
        data_specs = model_config.data_specs
        position_dim = data_specs.position_dim

        self.data_specs = data_specs
        self.use_rope = model_config.transformer_block_config.use_rope

        # Position encoding:
        self.pos_embed = ContinuousSincosEmbed(
            config=ContinuousSincosEmbeddingConfig(hidden_dim=hidden_dim, input_dim=position_dim),
        )

        # Optional RoPE:
        if self.use_rope:
            self.rope = RopeFrequency(
                config=RopeFrequencyConfig(
                    hidden_dim=hidden_dim // model_config.transformer_block_config.num_heads,
                    input_dim=position_dim,
                    implementation="complex",
                ),
            )

        # Surface/volume bias MLPs:
        self.surface_bias = MLP(config=MLPConfig(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim))
        self.volume_bias = MLP(config=MLPConfig(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim))

        # Optional physics feature projections:
        self.use_physics_features = data_specs.use_physics_features
        if self.use_physics_features:
            if data_specs.surface_feature_dim_total > 0:
                self.project_surface_features = LinearProjection(
                    config=LinearProjectionConfig(
                        input_dim=data_specs.surface_feature_dim_total,
                        output_dim=hidden_dim,
                        init_weights="truncnormal002",
                    ),
                )
            if data_specs.volume_feature_dim_total > 0:
                self.project_volume_features = LinearProjection(
                    config=LinearProjectionConfig(
                        input_dim=data_specs.volume_feature_dim_total,
                        output_dim=hidden_dim,
                        init_weights="truncnormal002",
                    ),
                )

        # Backbone
        self.backbone = Transformer(config=model_config)

        # Output projection
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.out = LinearProjection(
            config=LinearProjectionConfig(
                input_dim=hidden_dim, output_dim=data_specs.total_output_dim, init_weights="truncnormal002"
            ),
        )

    def forward(
        self,
        surface_position: torch.Tensor,
        volume_position: torch.Tensor,
        surface_features: torch.Tensor | None = None,
        volume_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        num_surface = surface_position.shape[1]
        input_position = torch.cat([surface_position, volume_position], dim=1)

        attn_kwargs: dict[str, torch.Tensor] = {}
        if self.use_rope:
            attn_kwargs["freqs"] = self.rope(input_position)

        x = self.pos_embed(input_position)

        if self.use_physics_features:
            parts = []
            if surface_features is not None and hasattr(self, "project_surface_features"):
                parts.append(self.project_surface_features(surface_features))
            if volume_features is not None and hasattr(self, "project_volume_features"):
                parts.append(self.project_volume_features(volume_features))
            if parts:
                x = x + torch.cat(parts, dim=1)

        # Surface/volume bias
        x_surface = self.surface_bias(x[:, :num_surface])
        x_volume = self.volume_bias(x[:, num_surface:])
        x = torch.cat([x_surface, x_volume], dim=1)

        x = self.backbone(x=x, attn_kwargs=attn_kwargs)
        x = self.out(self.norm(x))

        return self._gather_outputs(x, num_surface)

    def _gather_outputs(self, x: torch.Tensor, num_surface: int) -> dict[str, torch.Tensor]:
        """Split the flat output tensor into per-field dicts using data_specs."""
        surface_out = x[:, :num_surface]
        volume_out = x[:, num_surface:]
        result: dict[str, torch.Tensor] = {}

        offset = 0
        for name, dim in self.data_specs.surface_output_dims.items():
            result[f"surface_{name}"] = surface_out[..., offset : offset + dim]
            offset += dim

        if self.data_specs.volume_output_dims is not None:
            offset = 0
            for name, dim in self.data_specs.volume_output_dims.items():
                result[f"volume_{name}"] = volume_out[..., offset : offset + dim]
                offset += dim

        return result
