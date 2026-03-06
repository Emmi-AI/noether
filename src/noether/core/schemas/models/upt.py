#  Copyright © 2025 Emmi AI GmbH. All rights reserved.


from typing import Annotated

from pydantic import ConfigDict, Field, model_validator

from noether.core.schemas.dataset import AeroDataSpecs
from noether.core.schemas.mixins import InjectSharedFieldFromParentMixin, Shared
from noether.core.schemas.modules import DeepPerceiverDecoderConfig, SupernodePoolingConfig
from noether.core.schemas.modules.blocks import TransformerBlockConfig
from noether.core.schemas.modules.layers import (
    ContinuousSincosEmbeddingConfig,
    LinearProjectionConfig,
    RopeFrequencyConfig,
)

from .base import ModelBaseConfig


class UPTConfig(ModelBaseConfig, InjectSharedFieldFromParentMixin):
    """Configuration for a UPT model."""

    model_config = ConfigDict(extra="forbid")

    num_heads: int = Field(..., ge=1)
    """Number of attention heads in the model."""

    hidden_dim: int = Field(..., ge=1)
    """Hidden dimension of the model."""

    mlp_expansion_factor: int = Field(..., ge=1)
    """Expansion factor for the MLP of the FF layers."""

    approximator_depth: int = Field(..., ge=1)
    """Number of approximator layers."""

    use_rope: bool = Field(False)

    supernode_pooling_config: Annotated[SupernodePoolingConfig, Shared]

    approximator_config: Annotated[TransformerBlockConfig, Shared]

    decoder_config: Annotated[DeepPerceiverDecoderConfig, Shared]

    bias_layers: bool = Field(False)

    data_specs: AeroDataSpecs

    pos_embedding_config: ContinuousSincosEmbeddingConfig | None = None

    rope_frequency_config: RopeFrequencyConfig | None = None

    linear_projection_config: LinearProjectionConfig | None = None

    @model_validator(mode="after")
    def set_linear_projection_config(self) -> "UPTConfig":
        """Set the input_dim of the LinearProjectionConfig to match the hidden_dim of the model and the output_dim to match the total_output_dim of the data_specs."""
        if self.linear_projection_config is None:
            self.linear_projection_config = LinearProjectionConfig(
                input_dim=self.hidden_dim,
                output_dim=self.data_specs.total_output_dim,
                init_weights=self.decoder_config.perceiver_block_config.init_weights,
            )
        return self

    @model_validator(mode="after")
    def set_rope_frequency_config(self) -> "UPTConfig":
        """If use_rope is True, set the hidden_dim and input_dim of the RopeFrequencyConfig to match the model's hidden_dim and data_specs.position_dim."""
        if self.use_rope:
            self.rope_frequency_config = RopeFrequencyConfig(
                hidden_dim=self.hidden_dim // self.num_heads,
                input_dim=self.data_specs.position_dim,
                implementation="complex",
            )
        return self

    @model_validator(mode="after")
    def update_supernode_pooling_config(self) -> "UPTConfig":
        """Inject shared fields into supernode_pooling_config."""
        if self.data_specs.use_physics_features:
            self.supernode_pooling_config.input_features_dim = self.data_specs.surface_feature_dim_total
        return self

    @model_validator(mode="after")
    def set_sincos_embedding_config(self) -> "UPTConfig":
        # TODO: check if we can set this sucht that it cannot be configured via YAML
        """Set the hidden_dim of the ContinuousSincosEmbeddingConfig to match the model's hidden_dim."""
        if self.pos_embedding_config is None:
            self.pos_embedding_config = ContinuousSincosEmbeddingConfig(
                hidden_dim=self.hidden_dim,
                input_dim=self.data_specs.position_dim,
            )
        return self

    @model_validator(mode="after")
    def validate_parameters(self) -> "UPTConfig":
        """Validate validity of parameters across the model and its submodules.

        Ensures that:
        1. hidden_dim is divisible by num_heads in parent and all submodules with num_heads
        2. hidden_dim is consistent across parent and all submodules
        """
        # 1. Parent check: hidden_dim % num_heads == 0
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads}).")

        # 2. SupernodePoolingConfig: hidden_dim equality
        if self.supernode_pooling_config.hidden_dim != self.hidden_dim:
            raise ValueError(
                f"supernode_pooling_config.hidden_dim ({self.supernode_pooling_config.hidden_dim}) "
                f"must match model hidden_dim ({self.hidden_dim})."
            )

        # 3. ApproximatorConfig: hidden_dim equality + modulo check
        if self.approximator_config.hidden_dim != self.hidden_dim:
            raise ValueError(
                f"approximator_config.hidden_dim ({self.approximator_config.hidden_dim}) "
                f"must match model hidden_dim ({self.hidden_dim})."
            )

        if self.approximator_config.hidden_dim % self.approximator_config.num_heads != 0:
            raise ValueError(
                f"approximator_config.hidden_dim ({self.approximator_config.hidden_dim}) "
                f"must be divisible by approximator_config.num_heads ({self.approximator_config.num_heads})."
            )

        # 4. DecoderConfig: check nested perceiver_block_config
        perceiver_config = self.decoder_config.perceiver_block_config

        if perceiver_config.hidden_dim != self.hidden_dim:
            raise ValueError(
                f"decoder_config.perceiver_block_config.hidden_dim ({perceiver_config.hidden_dim}) "
                f"must match model hidden_dim ({self.hidden_dim})."
            )

        if perceiver_config.hidden_dim % perceiver_config.num_heads != 0:
            raise ValueError(
                f"decoder_config.perceiver_block_config.hidden_dim ({perceiver_config.hidden_dim}) "
                f"must be divisible by decoder_config.perceiver_block_config.num_heads ({perceiver_config.num_heads})."
            )

        return self
