#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import torch
import torch.nn as nn

from noether.core.models import Model
from noether.core.schemas.dataset import AeroDataSpecs
from noether.core.schemas.models import (
    AnchorBranchedUPTConfig,
    TransformerConfig,
    TransolverConfig,
    UPTConfig,
)
from noether.core.schemas.modules.layers import (
    ContinuousSincosEmbeddingConfig,
    LinearProjectionConfig,
    RopeFrequencyConfig,
)
from noether.core.schemas.modules.mlp import MLPConfig
from noether.modeling.models.ab_upt import AnchoredBranchedUPT
from noether.modeling.models.transformer import Transformer
from noether.modeling.models.upt import UPT
from noether.modeling.modules.layers import ContinuousSincosEmbed, LinearProjection, RopeFrequency
from noether.modeling.modules.mlp import MLP


class AeroTransformerConfig(TransformerConfig):
    """Transformer config extended with aerodynamic data specifications."""

    data_specs: AeroDataSpecs


class AeroTransolverConfig(TransolverConfig):
    """Transolver config extended with aerodynamic data specifications."""

    data_specs: AeroDataSpecs


def _gather_outputs(
    x: torch.Tensor,
    num_surface: int,
    data_specs: AeroDataSpecs,
) -> dict[str, torch.Tensor]:
    """Split a flat prediction tensor into named surface/volume output fields."""
    surface_out = x[:, :num_surface]
    volume_out = x[:, num_surface:]
    result: dict[str, torch.Tensor] = {}

    offset = 0
    for name, dim in data_specs.surface_output_dims.items():
        result[f"surface_{name}"] = surface_out[..., offset : offset + dim]
        offset += dim

    if data_specs.volume_output_dims is not None:
        offset = 0
        for name, dim in data_specs.volume_output_dims.items():
            result[f"volume_{name}"] = volume_out[..., offset : offset + dim]
            offset += dim

    return result


class AeroTransformer(Model):
    """Aerodynamic Transformer wrapper.

    End-to-end forward for aero CFD: positional encoding, optional RoPE, optional physics features,
    surface/volume bias, Transformer backbone, output projection, and output gathering.
    """

    def __init__(self, model_config: AeroTransformerConfig, **kwargs):
        super().__init__(model_config=model_config, **kwargs)

        hidden_dim = model_config.hidden_dim
        data_specs = model_config.data_specs
        position_dim = data_specs.position_dim

        self.data_specs = data_specs
        self.use_rope = model_config.transformer_block_config.use_rope

        self.pos_embed = ContinuousSincosEmbed(
            config=ContinuousSincosEmbeddingConfig(hidden_dim=hidden_dim, input_dim=position_dim),
        )

        if self.use_rope:
            self.rope = RopeFrequency(
                config=RopeFrequencyConfig(
                    hidden_dim=hidden_dim // model_config.transformer_block_config.num_heads,
                    input_dim=position_dim,
                    implementation="complex",
                ),
            )

        self.surface_bias = MLP(config=MLPConfig(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim))
        self.volume_bias = MLP(config=MLPConfig(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim))

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

        self.backbone = Transformer(config=model_config)

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
            parts: list[torch.Tensor] = []
            if surface_features is not None and hasattr(self, "project_surface_features"):
                parts.append(self.project_surface_features(surface_features))
            if volume_features is not None and hasattr(self, "project_volume_features"):
                parts.append(self.project_volume_features(volume_features))
            if parts:
                x = x + torch.cat(parts, dim=1)

        x_surface = self.surface_bias(x[:, :num_surface])
        x_volume = self.volume_bias(x[:, num_surface:])
        x = torch.cat([x_surface, x_volume], dim=1)

        x = self.backbone(x=x, attn_kwargs=attn_kwargs)
        x = self.out(self.norm(x))

        return _gather_outputs(x, num_surface, self.data_specs)


class AeroTransolver(Model):
    """Aerodynamic Transolver wrapper.

    Like ``AeroTransformer`` but adds the Transolver-specific learnable placeholder parameter.
    """

    def __init__(self, model_config: AeroTransolverConfig, **kwargs):
        super().__init__(model_config=model_config, **kwargs)

        hidden_dim = model_config.hidden_dim
        data_specs = model_config.data_specs
        position_dim = data_specs.position_dim

        self.data_specs = data_specs

        self.pos_embed = ContinuousSincosEmbed(
            config=ContinuousSincosEmbeddingConfig(hidden_dim=hidden_dim, input_dim=position_dim),
        )

        self.surface_bias = MLP(config=MLPConfig(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim))
        self.volume_bias = MLP(config=MLPConfig(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim))

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

        self.placeholder = nn.Parameter(torch.rand(1, 1, hidden_dim) / hidden_dim)

        self.backbone = Transformer(config=model_config)

        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.out = LinearProjection(
            config=LinearProjectionConfig(
                input_dim=hidden_dim,
                output_dim=data_specs.total_output_dim,
                init_weights="truncnormal002",
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

        x = self.pos_embed(input_position)

        if self.use_physics_features:
            parts: list[torch.Tensor] = []
            if surface_features is not None and hasattr(self, "project_surface_features"):
                parts.append(self.project_surface_features(surface_features))
            if volume_features is not None and hasattr(self, "project_volume_features"):
                parts.append(self.project_volume_features(volume_features))
            if parts:
                x = x + torch.cat(parts, dim=1)

        x_surface = self.surface_bias(x[:, :num_surface])
        x_volume = self.volume_bias(x[:, num_surface:])
        x = torch.cat([x_surface, x_volume], dim=1)

        x = x + self.placeholder

        x = self.backbone(x=x, attn_kwargs={})
        x = self.out(self.norm(x))

        return _gather_outputs(x, num_surface, self.data_specs)


class AeroUPT(Model):
    """Aerodynamic UPT wrapper.

    Combines separate surface/volume query positions into the single ``query_position``
    that the core UPT expects, and splits outputs using ``AeroDataSpecs``.
    """

    def __init__(self, model_config: UPTConfig, **kwargs):
        super().__init__(model_config=model_config, **kwargs)
        self.backbone = UPT(config=model_config)
        self.data_specs = model_config.data_specs

    def forward(
        self,
        surface_position_batch_idx: torch.Tensor,
        surface_position_supernode_idx: torch.Tensor,
        surface_position: torch.Tensor,
        surface_query_position: torch.Tensor,
        volume_query_position: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        query_position = torch.cat([surface_query_position, volume_query_position], dim=1)

        x = self.backbone(
            surface_position_batch_idx=surface_position_batch_idx,
            surface_position_supernode_idx=surface_position_supernode_idx,
            surface_position=surface_position,
            query_position=query_position,
        )

        num_surface = surface_query_position.shape[1]
        return _gather_outputs(x, num_surface, self.data_specs)


class AeroABUPT(Model):
    """Aerodynamic Anchored-Branched UPT wrapper.

    Bridges the factory's ``(config, **kwargs)`` instantiation pattern to the core model.
    """

    def __init__(self, model_config: AnchorBranchedUPTConfig, **kwargs) -> None:
        super().__init__(model_config=model_config, **kwargs)
        self.backbone = AnchoredBranchedUPT(config=model_config)

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        return self.backbone(**kwargs)  # type: ignore[no-any-return]
