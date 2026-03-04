#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from noether.core.types import InitWeightsMode


class TransformerBlockConfig(BaseModel):
    """Configuration for a transformer block."""

    hidden_dim: int = Field(..., ge=1)
    """Hidden Dimension of the transformer block."""

    num_heads: int = Field(..., ge=1)
    """Number of attention heads."""

    mlp_hidden_dim: int | None = Field(None)
    """Hidden dimension of the MLP layer. If set to None, the mlp_hidden dim is set to hidden_dim * mlp_expansion_factor in the TransformerConfig. If both are None, an error is raised."""

    mlp_expansion_factor: int | None = Field(None, ge=1)
    """Expansion factor for the MLP hidden dimension relative to the hidden dimension. If 'mlp_hidden_dim' is not set, this factor is used to compute it as hidden_dim * mlp_expansion_factor."""

    drop_path: float = Field(0.0, ge=0.0, le=1.0)
    """Probability to drop the attention or MLP module. Defaults to 0.0."""

    attention_constructor: Literal[
        "dot_product",
        "perceiver",
        "transolver",
        "transolver_plusplus",
    ] = "dot_product"
    """Constructor of the attention module. Defaults to 'dot_product'."""

    layerscale: float | None = Field(None, ge=0.0)
    """ Init scale value to scale layer activations. Defaults to None."""

    condition_dim: int | None = Field(None)
    """Dimension of the conditioning vector. If none, no conditioning is applied. If provided, the transformer block will turn into a Diffusion Transformer (DiT) block."""

    bias: bool = Field(True)
    """Whether to use biases in norm/projections. Defaults to True."""

    eps: float = Field(1e-6, gt=0.0)
    """Epsilon Value for the layer nornalization. Defaults to 1e-6."""

    init_weights: InitWeightsMode = Field("truncnormal002")
    """Initialization method for the weight matrices of the network. Defaults to "truncnormal002"""

    use_rope: bool = Field(False)
    """Whether to use Rotary Positional Embeddings (RoPE)."""

    attention_arguments: dict = {}
    """Additional arguments for the attention module that are only needed for a specific attention implementation."""

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


class PerceiverBlockConfig(TransformerBlockConfig):
    """Configuration for the PerceiverBlock module."""

    kv_dim: int | None = Field(None)
    """Dimensionality of the key and value representations. Defaults to None. If None, hidden_dim is used."""

    @model_validator(mode="after")
    def set_kv_dim(self) -> "PerceiverBlockConfig":
        """Set kv_dim to hidden_dim if not provided."""
        if self.kv_dim is None and self.condition_dim is None:
            self.kv_dim = self.hidden_dim
        return self
