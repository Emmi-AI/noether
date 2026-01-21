#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from pydantic import ConfigDict, model_validator

from noether.core.schemas.modules.blocks import TransformerBlockConfig

from .base import ModelBaseConfig


class TransformerConfig(TransformerBlockConfig, ModelBaseConfig):
    """Configuration for a Transformer model."""

    model_config = ConfigDict(extra="forbid")

    depth: int
    """Number of transformer blocks in the model."""
    mlp_expansion_factor: int = 4
    """Expansion factor for the MLP hidden dimension relative to the hidden dimension. If 'mlp_hidden_dim' is not set, this factor is used to compute it as hidden_dim * mlp_expansion_factor."""

    @model_validator(mode="after")
    def set_mlp_hidden_dim(self):
        if self.mlp_hidden_dim is None:
            if self.mlp_expansion_factor is None:
                raise ValueError("Either 'mlp_hidden_dim' or 'mlp_expansion_factor' must be provided.")
            self.mlp_hidden_dim = self.hidden_dim * self.mlp_expansion_factor
        return self
