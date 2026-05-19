#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import torch
import torch.nn as nn

from noether.core.models.model import Model
from noether.data.schemas import ModelDataSpecs
from noether.modeling.models.ab_upt import AnchorBranchedUPTConfig, AnchoredBranchedUPT
from noether.modeling.models.transformer import Transformer, TransformerConfig
from noether.modeling.models.transolver import TransolverConfig
from noether.modeling.models.upt import UPT, UPTConfig
from noether.modeling.modules.layers import ContinuousSincosEmbed, LinearProjection, RopeFrequency
from noether.modeling.modules.layers.continuous_sincos_embed import ContinuousSincosEmbeddingConfig
from noether.modeling.modules.layers.linear_projection import LinearProjectionConfig
from noether.modeling.modules.layers.rope_frequency import RopeFrequencyConfig
from noether.modeling.modules.mlp import MLP, MLPConfig


class AeroTransformerConfig(TransformerConfig):
    """Transformer config extended with aerodynamic data specifications."""

    data_specs: ModelDataSpecs


class AeroTransolverConfig(TransolverConfig):
    """Transolver config extended with aerodynamic data specifications."""

    data_specs: ModelDataSpecs


def _domain_feature_dim(data_specs: ModelDataSpecs, domain: str) -> int:
    """Return total feature dim for a domain, or 0 if absent."""
    spec = data_specs.domains.get(domain)
    if spec and spec.feature_dim:
        return spec.feature_dim.total_dim
    return 0


def _gather_outputs(
    x: torch.Tensor,
    num_surface: int,
    data_specs: ModelDataSpecs,
) -> dict[str, torch.Tensor]:
    """Split a flat prediction tensor into named surface/volume output fields."""
    surface_out = x[:, :num_surface]
    volume_out = x[:, num_surface:]
    result: dict[str, torch.Tensor] = {}

    surface_spec = data_specs.domains.get("surface")
    if surface_spec:
        offset = 0
        for name, dim in surface_spec.output_dims.items():
            result[f"surface_{name}"] = surface_out[..., offset : offset + dim]
            offset += dim

    volume_spec = data_specs.domains.get("volume")
    if volume_spec:
        offset = 0
        for name, dim in volume_spec.output_dims.items():
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
            config=ContinuousSincosEmbeddingConfig(
                hidden_dim=hidden_dim,
                input_dim=position_dim,
                max_wavelength=model_config.transformer_block_config.max_wavelength,
            ),
        )

        if self.use_rope:
            self.rope = RopeFrequency(
                config=RopeFrequencyConfig(
                    hidden_dim=hidden_dim // model_config.transformer_block_config.num_heads,
                    input_dim=position_dim,
                    implementation="complex",
                    max_wavelength=model_config.transformer_block_config.max_wavelength,
                ),
            )

        self.surface_bias = MLP(
            config=MLPConfig(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                bias=model_config.transformer_block_config.bias,
            )
        )
        self.volume_bias = MLP(
            config=MLPConfig(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                bias=model_config.transformer_block_config.bias,
            )
        )

        self.use_physics_features = data_specs.use_physics_features
        if self.use_physics_features:
            surface_feat_dim = _domain_feature_dim(data_specs, "surface")
            if surface_feat_dim > 0:
                self.project_surface_features = LinearProjection(
                    config=LinearProjectionConfig(
                        input_dim=surface_feat_dim,
                        output_dim=hidden_dim,
                        init_weights="truncnormal002",
                        bias=model_config.transformer_block_config.bias,
                    ),
                )
            volume_feat_dim = _domain_feature_dim(data_specs, "volume")
            if volume_feat_dim > 0:
                self.project_volume_features = LinearProjection(
                    config=LinearProjectionConfig(
                        input_dim=volume_feat_dim,
                        output_dim=hidden_dim,
                        init_weights="truncnormal002",
                        bias=model_config.transformer_block_config.bias,
                    ),
                )

        self.backbone = Transformer(config=model_config)

        self.norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.out = LinearProjection(
            config=LinearProjectionConfig(
                input_dim=hidden_dim,
                output_dim=data_specs.total_output_dim,
                init_weights="truncnormal002",
                bias=model_config.transformer_block_config.bias,
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

        self.surface_bias = MLP(
            config=MLPConfig(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                bias=model_config.transformer_block_config.bias,
            )
        )
        self.volume_bias = MLP(
            config=MLPConfig(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                bias=model_config.transformer_block_config.bias,
            )
        )

        self.use_physics_features = data_specs.use_physics_features
        if self.use_physics_features:
            surface_feat_dim = _domain_feature_dim(data_specs, "surface")
            if surface_feat_dim > 0:
                self.project_surface_features = LinearProjection(
                    config=LinearProjectionConfig(
                        input_dim=surface_feat_dim,
                        output_dim=hidden_dim,
                        init_weights="truncnormal002",
                        bias=model_config.transformer_block_config.bias,
                    ),
                )
            volume_feat_dim = _domain_feature_dim(data_specs, "volume")
            if volume_feat_dim > 0:
                self.project_volume_features = LinearProjection(
                    config=LinearProjectionConfig(
                        input_dim=volume_feat_dim,
                        output_dim=hidden_dim,
                        init_weights="truncnormal002",
                        bias=model_config.transformer_block_config.bias,
                    ),
                )

        self.placeholder = nn.Parameter(torch.rand(1, 1, hidden_dim) / hidden_dim)

        self.backbone = Transformer(config=model_config)

        self.norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.out = LinearProjection(
            config=LinearProjectionConfig(
                input_dim=hidden_dim,
                output_dim=data_specs.total_output_dim,
                init_weights="truncnormal002",
                bias=model_config.transformer_block_config.bias,
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
    that the core UPT expects, and splits outputs using ``ModelDataSpecs``.
    Supports optional surface/volume bias layers and physics feature projection on queries.
    """

    def __init__(self, model_config: UPTConfig, **kwargs):
        super().__init__(model_config=model_config, **kwargs)
        self.backbone = UPT(config=model_config)
        self.data_specs = model_config.data_specs

        hidden_dim = model_config.hidden_dim
        self.use_bias_layers = model_config.bias_layers
        if self.use_bias_layers:
            self.surface_bias = MLP(
                config=MLPConfig(
                    input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, bias=model_config.bias
                )
            )
            self.volume_bias = MLP(
                config=MLPConfig(
                    input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, bias=model_config.bias
                )
            )

        self.use_physics_features = model_config.data_specs.use_physics_features
        if self.use_physics_features:
            surface_feat_dim = _domain_feature_dim(model_config.data_specs, "surface")
            if surface_feat_dim > 0:
                self.project_surface_features = LinearProjection(
                    config=LinearProjectionConfig(
                        input_dim=surface_feat_dim,
                        output_dim=hidden_dim,
                        init_weights="truncnormal002",
                        bias=model_config.bias,
                    ),
                )
            volume_feat_dim = _domain_feature_dim(model_config.data_specs, "volume")
            if volume_feat_dim > 0:
                self.project_volume_features = LinearProjection(
                    config=LinearProjectionConfig(
                        input_dim=volume_feat_dim,
                        output_dim=hidden_dim,
                        init_weights="truncnormal002",
                        bias=model_config.bias,
                    ),
                )

    def forward(
        self,
        surface_position_batch_idx: torch.Tensor,
        surface_position_supernode_idx: torch.Tensor,
        surface_position: torch.Tensor,
        surface_query_position: torch.Tensor,
        volume_query_position: torch.Tensor,
        surface_query_features: torch.Tensor | None = None,
        volume_query_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        num_surface = surface_query_position.shape[1]
        query_position = torch.cat([surface_query_position, volume_query_position], dim=1)

        encoder_attn_kwargs, decoder_attn_kwargs = self.backbone.compute_rope_args(
            surface_position_batch_idx, surface_position, surface_position_supernode_idx, query_position
        )

        # Supernode pooling encoder:
        x = self.backbone.encoder(
            input_pos=surface_position,
            supernode_idx=surface_position_supernode_idx,
            batch_idx=surface_position_batch_idx,
        )
        # Approximator blocks:
        for block in self.backbone.approximator_blocks:
            x, _ = block(x, attn_kwargs=encoder_attn_kwargs)

        # Query embeddings with optional bias and physics features:
        queries = self.backbone.pos_embed(query_position)

        if self.use_bias_layers:
            q_surface = self.surface_bias(queries[:, :num_surface])
            q_volume = self.volume_bias(queries[:, num_surface:])
            queries = torch.cat([q_surface, q_volume], dim=1)

        if self.use_physics_features:
            parts: list[torch.Tensor] = []
            if surface_query_features is not None and hasattr(self, "project_surface_features"):
                parts.append(self.project_surface_features(surface_query_features))
            if volume_query_features is not None and hasattr(self, "project_volume_features"):
                parts.append(self.project_volume_features(volume_query_features))
            if parts:
                queries = queries + torch.cat(parts, dim=1)

        # Perceiver decoder:
        x = self.backbone.decoder(kv=x, queries=queries, attn_kwargs=decoder_attn_kwargs, condition=None)

        x = self.backbone.norm(x)
        x = self.backbone.prediction_layer(x)

        return _gather_outputs(x, num_surface, self.data_specs)


class AeroABUPT(Model):
    """Aerodynamic Anchored-Branched UPT wrapper.

    Bridges the factory's ``(config, **kwargs)`` instantiation pattern to the core model.
    Converts flat kwargs (``surface_anchor_position``, ``volume_anchor_position``, ...)
    into the domain-dict format expected by :class:`AnchoredBranchedUPT`.
    """

    def __init__(self, model_config: AnchorBranchedUPTConfig, **kwargs) -> None:
        super().__init__(model_config=model_config, **kwargs)
        self.backbone = AnchoredBranchedUPT(config=model_config)
        self._domain_names = list(model_config.data_specs.domains.keys())
        self._conditioning_keys = (
            list(model_config.data_specs.conditioning_dims.keys()) if model_config.data_specs.conditioning_dims else []
        )

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        domain_anchor_positions: dict[str, torch.Tensor] = {}
        domain_query_positions: dict[str, torch.Tensor] = {}
        conditioning_inputs: dict[str, torch.Tensor] = {}
        domain_anchor_features: dict[str, torch.Tensor] = {}
        domain_query_features: dict[str, torch.Tensor] = {}

        for name in self._domain_names:
            if f"{name}_anchor_position" in kwargs:
                domain_anchor_positions[name] = kwargs[f"{name}_anchor_position"]
            if f"query_{name}_position" in kwargs:
                domain_query_positions[name] = kwargs[f"query_{name}_position"]
            if f"{name}_anchor_features" in kwargs:
                domain_anchor_features[name] = kwargs[f"{name}_anchor_features"]
            if f"{name}_query_features" in kwargs:
                domain_query_features[name] = kwargs[f"{name}_query_features"]

        for key in self._conditioning_keys:
            if key in kwargs:
                conditioning_inputs[key] = kwargs[key]

        predictions, _ = self.backbone(
            geometry_position=kwargs.get("geometry_position"),
            geometry_supernode_idx=kwargs.get("geometry_supernode_idx"),
            geometry_batch_idx=kwargs.get("geometry_batch_idx"),
            domain_anchor_positions=domain_anchor_positions or None,
            domain_query_positions=domain_query_positions or None,
            conditioning_inputs=conditioning_inputs or None,
            domain_anchor_features=domain_anchor_features or None,
            domain_query_features=domain_query_features or None,
        )
        return predictions  # type: ignore[no-any-return]
