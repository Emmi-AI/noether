#  Copyright © 2025 Emmi AI GmbH. All rights reserved.


from typing import Annotated

from pydantic import ConfigDict, Field, model_validator

from noether.core.schemas.dataset import AeroDataSpecs
from noether.core.schemas.mixins import InjectSharedFieldFromParentMixin, Shared
from noether.core.schemas.modules import DeepPerceiverDecoderConfig, SupernodePoolingConfig
from noether.core.schemas.modules.blocks import TransformerBlockConfig

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
