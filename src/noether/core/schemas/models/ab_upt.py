#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import warnings
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, computed_field, model_validator

from noether.core.schemas.dataset import ModelDataSpecs
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

    supernode_pooling_config: Annotated[SupernodePoolingConfig, Shared] | None = None

    transformer_block_config: Annotated[TransformerBlockConfig, Shared]

    geometry_depth: int = Field(..., ge=0)
    """Number of transformer blocks in the geometry encoder."""

    hidden_dim: int = Field(..., ge=1)
    """Hidden dimension of the model."""

    physics_blocks: list[Literal["self", "shared", "cross", "joint", "perceiver"]]
    """Types of physics blocks to use in the model.
    Options are "self", "cross", "joint", and "perceiver".
    Self: Self-attention within a branch. Attention weights are shared between all domains.
    Cross: Cross-attention between domains. Each domain attends to all other domains' anchors.
    Joint: Joint attention over all domain points. Full self-attention over all points.
    Perceiver: Perceiver-style cross-attention to geometry encoding.

    Note: "shared" is a deprecated alias for "self" and will be removed in a future release."""

    num_domain_decoder_blocks: dict[str, int]
    """Number of final domain-specific decoder blocks with self attention and no weight sharing, e.g. {"surface": 2, "volume": 2}."""

    init_weights: InitWeightsMode = Field("truncnormal002")
    """Weight initialization of linear layers. Defaults to "truncnormal002"."""

    drop_path_rate: float = Field(0.0)
    """Drop path rate for stochastic depth. Defaults to 0.0 (no drop path)."""

    data_specs: ModelDataSpecs
    """Data specifications for the model."""

    @model_validator(mode="after")
    def migrate_shared_to_self(self) -> "AnchorBranchedUPTConfig":
        """Migrate deprecated 'shared' block type to 'self'."""
        if "shared" in self.physics_blocks:
            warnings.warn(
                'physics_blocks: "shared" is deprecated, use "self" instead. '
                '"shared" will be removed in a future release.',
                DeprecationWarning,
                stacklevel=2,
            )
            self.physics_blocks = ["self" if b == "shared" else b for b in self.physics_blocks]
        return self

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
            max_wavelength=self.transformer_block_config.max_wavelength,
        )

    @computed_field
    def pos_embed_config(self) -> ContinuousSincosEmbeddingConfig:
        return ContinuousSincosEmbeddingConfig(
            hidden_dim=self.hidden_dim,
            input_dim=self.data_specs.position_dim,
            max_wavelength=self.transformer_block_config.max_wavelength,
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
    def domain_decoder_configs(self) -> dict[str, LinearProjectionConfig]:
        """Per-domain decoder projection configs, keyed by domain name."""
        return {
            name: LinearProjectionConfig(
                input_dim=self.hidden_dim,
                output_dim=spec.output_dims.total_dim,
                init_weights="truncnormal002",
            )
            for name, spec in self.data_specs.domains.items()
        }

    @model_validator(mode="after")
    def validate_parameters(self) -> "AnchorBranchedUPTConfig":
        """Validate validity of parameters across the model and its submodules.

        Ensures that hidden_dim is consistent across parent and all submodules.
        Note: transformer_block_config validates hidden_dim % num_heads == 0 in its own validator.
        """
        # SupernodePoolingConfig: hidden_dim equality
        if self.supernode_pooling_config is not None and self.supernode_pooling_config.hidden_dim != self.hidden_dim:
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
