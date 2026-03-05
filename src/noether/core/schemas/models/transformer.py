#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from pydantic import ConfigDict, Field, model_validator

from noether.core.schemas.modules.blocks import TransformerBlockConfig

from .base import ModelBaseConfig


class TransformerConfig(TransformerBlockConfig, ModelBaseConfig):
    """Configuration for a Transformer model."""

    model_config = ConfigDict(extra="forbid")

    depth: int = Field(..., ge=1)
    """Number of transformer blocks in the model."""
    mlp_expansion_factor: int = Field(4, ge=1)
    """Expansion factor for the MLP hidden dimension relative to the hidden dimension. If 'mlp_hidden_dim' is not set, this factor is used to compute it as hidden_dim * mlp_expansion_factor."""

    transformer_block_config: TransformerBlockConfig | None = None
    """Optional pre-defined configuration for the transformer blocks. If not provided, it will be automatically created from the parent config parameters."""

    @model_validator(mode="after")
    def set_mlp_hidden_dim(self):
        # Validate hidden_dim is divisible by num_heads
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads}).")

        if self.mlp_hidden_dim is None:
            if self.mlp_expansion_factor is None:
                raise ValueError("Either 'mlp_hidden_dim' or 'mlp_expansion_factor' must be provided.")
            self.mlp_hidden_dim = self.hidden_dim * self.mlp_expansion_factor

        return self

    @model_validator(mode="after")
    def create_transformer_block_config(self):
        if self.transformer_block_config is not None:
            return self
        block_fields = TransformerBlockConfig.model_fields.keys()
        self.transformer_block_config = TransformerBlockConfig(**{k: getattr(self, k) for k in block_fields})
        return self
