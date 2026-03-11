#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, computed_field, model_validator

from noether.core.schemas.dataset import AeroDataSpecs
from noether.core.schemas.mixins import InjectSharedFieldFromParentMixin, Shared
from noether.core.schemas.modules.blocks import PerceiverBlockConfig, TransformerBlockConfig
from noether.core.schemas.modules.encoders import SupernodePoolingConfig
from noether.core.schemas.modules.layers import (
    ContinuousSincosEmbeddingConfig,
    LinearProjectionConfig,
    RopeFrequencyConfig,
)
from noether.core.schemas.modules.mlp import MLPConfig
from noether.core.types import InitWeightsMode

from .base import ModelBaseConfig


class AnchorBranchedUPTConfig(ModelBaseConfig, InjectSharedFieldFromParentMixin):
    model_config = ConfigDict(extra="forbid")

    supernode_pooling_config: Annotated[SupernodePoolingConfig, Shared]

    transformer_block_config: Annotated[TransformerBlockConfig, Shared]

    geometry_depth: int = Field(..., ge=0)
    """Number of transformer blocks in the geometry encoder."""

    hidden_dim: int = Field(..., ge=1)
    """Hidden dimension of the model."""

    physics_blocks: list[Literal["shared", "cross", "joint", "perceiver"]]
    """Types of physics blocks to use in the model.
    Options are "shared", "cross", "joint", and "perceiver".
    Shared: Self-attention within a branch (surface or volume). Attention blocks share weights between surface and volume.
    Cross: Cross-attention between surface and volume branches. Weights are shared between surface and volume.
    Joint: Joint attention over surface and volume points. I.e. full self-attention over both surface and volume points.
    Perceiver: Perceiver-style attention blocks."""

    num_surface_blocks: int = Field(..., ge=1)
    """Number of transformer blocks in the surface decoder. Weights are not shared with the volume decoder."""

    num_volume_blocks: int = Field(..., ge=1)
    """Number of transformer blocks in the volume decoder. Weights are not shared with the surface decoder."""

    init_weights: InitWeightsMode = Field("truncnormal002")
    """Weight initialization of linear layers. Defaults to "truncnormal002"."""

    drop_path_rate: float = Field(0.0)
    """Drop path rate for stochastic depth. Defaults to 0.0 (no drop path)."""

    data_specs: AeroDataSpecs
    """Data specifications for the model."""

    @model_validator(mode="after")
    def set_condition_dim(self) -> "AnchorBranchedUPTConfig":
        """Set condition_dim in transformer_block_config based on data_specs."""

        if self.data_specs.conditioning_dims is not None and self.data_specs.conditioning_dims.total_dim > 0:
            condition_dim = self.data_specs.conditioning_dims.total_dim
        else:
            condition_dim = None
        self.transformer_block_config.condition_dim = condition_dim

        return self

    @computed_field
    def rope_frequency_config(self) -> RopeFrequencyConfig:
        return RopeFrequencyConfig(
            hidden_dim=self.transformer_block_config.hidden_dim // self.transformer_block_config.num_heads,
            input_dim=self.data_specs.position_dim,
            implementation="complex",
        )

    @computed_field
    def pos_embed_config(self) -> ContinuousSincosEmbeddingConfig:
        return ContinuousSincosEmbeddingConfig(
            hidden_dim=self.hidden_dim,
            input_dim=self.data_specs.position_dim,
        )

    @computed_field
    def bias_mlp_config(self) -> MLPConfig:
        return MLPConfig(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        )

    @computed_field
    def perceiver_block_config(self) -> PerceiverBlockConfig:
        return PerceiverBlockConfig(
            hidden_dim=self.hidden_dim,
            num_heads=self.transformer_block_config.num_heads,
            mlp_expansion_factor=self.transformer_block_config.mlp_expansion_factor,
            kv_dim=None,
            use_rope=self.transformer_block_config.use_rope,
            condition_dim=self.transformer_block_config.condition_dim,
        )

    @computed_field
    def surface_decoder_config(self) -> LinearProjectionConfig:
        return LinearProjectionConfig(
            input_dim=self.hidden_dim,
            output_dim=self.data_specs.surface_output_dims.total_dim,
            init_weights="truncnormal002",
        )

    @computed_field
    def volume_decoder_config(self) -> LinearProjectionConfig | None:
        if self.data_specs.volume_output_dims is None:
            return None
        return LinearProjectionConfig(
            input_dim=self.hidden_dim,
            output_dim=self.data_specs.volume_output_dims.total_dim,
            init_weights="truncnormal002",
        )

    @model_validator(mode="after")
    def validate_parameters(self) -> "AnchorBranchedUPTConfig":
        """Validate validity of parameters across the model and its submodules.

        Ensures that hidden_dim is consistent across parent and all submodules.
        Note: transformer_block_config validates hidden_dim % num_heads == 0 in its own validator.
        """
        # SupernodePoolingConfig: hidden_dim equality
        if self.supernode_pooling_config.hidden_dim != self.hidden_dim:
            raise ValueError(
                f"supernode_pooling_config.hidden_dim ({self.supernode_pooling_config.hidden_dim}) "
                f"must match model hidden_dim ({self.hidden_dim})."
            )

        # TransformerBlockConfig: hidden_dim equality
        if self.transformer_block_config.hidden_dim != self.hidden_dim:
            raise ValueError(
                f"transformer_block_config.hidden_dim ({self.transformer_block_config.hidden_dim}) "
                f"must match model hidden_dim ({self.hidden_dim})."
            )

        return self
