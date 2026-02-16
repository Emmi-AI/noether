#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import abc
from collections.abc import Sequence
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from noether.core.types import InitWeightsMode


# =====================================================================================================================
#                                                   REGULAR ATTENTION
# ---------------------------------------------------------------------------------------------------------------------
class AttentionConfig(BaseModel):
    """
    Configuration for an attention module.
    Since we can have many different attention implementations, we allow extra fields.
    such that we can use the same schema for all attention modules.
    """

    model_config = ConfigDict(extra="allow")

    """Configuration for an attention module."""

    hidden_dim: int = Field(..., ge=1)
    """Dimensionality of the hidden features."""

    num_heads: int = Field(..., ge=1)
    """Number of attention heads."""

    use_rope: bool = Field(False)
    """Whether to use Rotary Positional Embeddings (RoPE)."""

    dropout: float = Field(0.0, ge=0.0, le=1.0)
    """Dropout rate for the attention weights and output projection."""

    init_weights: InitWeightsMode = Field("truncnormal002")
    """Weight initialization strategy."""

    bias: bool = Field(True)
    """Whether to use bias terms in linear layers."""

    head_dim: int | None = Field(None)
    """Dimensionality of each attention head."""

    @model_validator(mode="after")
    def validate_hidden_dim_and_num_heads(self):
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("The 'hidden_dim' must be divisible by 'num_heads'.")
        self.head_dim = self.hidden_dim // self.num_heads
        return self


class DotProductAttentionConfig(AttentionConfig):
    """Configuration for the Dot Product attention module."""


class TransolverAttentionConfig(AttentionConfig):
    """Configuration for the Transolver attention module."""

    num_slices: int = Field(512)
    """Number of slices to project the input tokens to."""


class TransolverPlusPlusAttentionConfig(TransolverAttentionConfig):
    """Configuration for the Transolver++ attention module."""

    use_overparameterization: bool = Field(True)
    """Whether to use overparameterization for the slice projection."""

    use_adaptive_temperature: bool = Field(True)
    """Whether to use an adaptive temperature for the slice selection."""

    temperature_activation: Literal["sigmoid", "softplus", "exp"] | None = Field("softplus")
    """Activation function for the adaptive temperature."""

    use_gumbel_softmax: bool = Field(True)
    """Whether to use Gumbel-Softmax for the slice selection."""


class IrregularNatAttentionConfig(AttentionConfig):
    """Configuration for the Irregular Neighbourhood Attention Transformer (NAT) attention module."""

    input_dim: int = Field(..., ge=0)
    """Dimensionality of the input features."""

    radius: float = Field(0.1, ge=0.0)
    """Radius for the radius graph."""

    max_degree: int = Field(16, ge=1)
    """Maximum number of neighbors per point."""

    relpos_mlp_hidden_dim: int = Field(32, ge=1)
    """Hidden dimensionality of the relative position bias MLP."""

    relpos_mlp_dropout: float = Field(0.0, ge=0.0, le=1.0)
    """Dropout rate for the relative position bias MLP."""


class PerceiverAttentionConfig(AttentionConfig):
    """Configuration for the Perceiver attention module."""

    kv_dim: int | None = Field(None)
    """Dimensionality of the key/value features. If None, use hidden_dim."""

    @model_validator(mode="after")
    def set_kv_dim(self):
        if self.kv_dim is None:
            self.kv_dim = self.hidden_dim
        return self


# =====================================================================================================================


# =====================================================================================================================
#                                                   ANCHOR ATTENTION
# ---------------------------------------------------------------------------------------------------------------------
class MultiBranchAnchorAttentionConfig(AttentionConfig, metaclass=abc.ABCMeta):
    """Configuration for Multi-Branch Anchor Attention module."""

    branches: list[str] = Field(..., min_length=1)
    anchor_suffix: str = Field("_anchors")


class CrossAnchorAttentionConfig(MultiBranchAnchorAttentionConfig):
    """Configuration for Cross Anchor Attention module."""


class JointAnchorAttentionConfig(MultiBranchAnchorAttentionConfig):
    """Configuration for Joint Anchor Attention module."""


class TokenSpec(BaseModel):
    """Specification for a token type in the attention mechanism."""

    name: Literal[
        "surface_anchors", "volume_anchors", "surface_queries", "volume_queries"
    ]  # Semantic identifier (e.g., "surface_anchors")
    size: int = Field(..., ge=0)  # Number of tokens of this type (i.e. the sequence length)

    @classmethod
    def from_dict(cls, token_dict: dict[str, int]) -> "TokenSpec":
        """Create TokenSpec from dictionary with single key-value pair."""
        if len(token_dict) != 1:
            raise ValueError("Dictionary must contain exactly one key-value pair")
        name, size = next(iter(token_dict.items()))
        valid_name = cast("Literal['surface_anchors', 'volume_anchors', 'surface_queries', 'volume_queries']", name)
        return cls(name=valid_name, size=size)

    def to_dict(self) -> dict[str, int]:
        """Convert TokenSpec to dictionary."""
        return {self.name: self.size}


class AttentionPattern(BaseModel):
    """Defines which tokens attend to which other tokens."""

    query_tokens: Sequence[str]  #  The tokens that attend to the key/value tokens, e.g. ["anchors", "queries"]
    key_value_tokens: Sequence[str]  # The tokens that are attended to by the query tokens, e.g. ["anchors"]


class MixedAttentionConfig(DotProductAttentionConfig):
    """Configuration for Mixed Attention module."""


# =====================================================================================================================
