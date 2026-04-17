#  Copyright © 2025 Emmi AI GmbH. All rights reserved.


from pydantic import ConfigDict, model_validator

from noether.core.schemas.models.transformer import TransformerConfig

from .base import ModelBaseConfig


class TransolverConfig(TransformerConfig, ModelBaseConfig):
    """Configuration for a Transolver model."""

    model_config = ConfigDict(extra="forbid")

    attention_arguments: dict = {"num_slices": 512}  # test if this can be overwritten in the model config

    @model_validator(mode="after")
    def set_attention_constructor(self) -> "TransolverConfig":
        """Set attention_constructor in transformer_block_config based on data_specs."""
        self.transformer_block_config.attention_constructor = "transolver"  # type: ignore[assignment]

        return self


class TransolverPlusPlusConfig(TransolverConfig):
    """Configuration for a Transolver++ model."""

    @model_validator(mode="after")
    def set_attention_constructor(self) -> "TransolverPlusPlusConfig":
        """Set attention_constructor in transformer_block_config based on data_specs."""
        self.transformer_block_config.attention_constructor = "transolver_plusplus"  # type: ignore[assignment]

        return self
