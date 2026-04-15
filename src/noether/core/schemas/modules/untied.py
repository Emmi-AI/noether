#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from pydantic import BaseModel, Field, computed_field

from noether.core.schemas.modules.attention import MixedAttentionConfig
from noether.core.schemas.modules.blocks import PerceiverBlockConfig, TransformerBlockConfig
from noether.core.schemas.modules.layers import LinearProjectionConfig
from noether.core.schemas.modules.mlp import MLPConfig


class UntiedLinearConfig(BaseModel):
    """Configuration for a linear layer with per-type (untied) weight banks.

    Composes a :class:`LinearProjectionConfig` (shared across types) with a
    ``num_types`` field: each token type gets its own independent weight matrix
    with the geometry described by the linear projection config.
    """

    num_types: int = Field(..., ge=1)
    """Number of distinct token types, each with its own weight bank."""

    linear_projection: LinearProjectionConfig
    """Shared geometry (input/output dims, bias, init) for every per-type weight bank."""


class UntiedMixedAttentionConfig(MixedAttentionConfig):
    """Configuration for multi-head attention with per-type (untied) QKV and output projections.

    Extends :class:`MixedAttentionConfig` with a ``num_types`` field: the QKV and
    output projections are :class:`UntiedLinear` layers so each token type gets
    its own projection weights. Attention itself is still computed across all
    tokens via :meth:`MixedAttention._process_pattern_batched`.
    """

    num_types: int = Field(..., ge=1)
    """Number of distinct token types, each with its own QKV/output weight bank."""

    @computed_field
    def projection_config(self) -> UntiedLinearConfig:
        """Configuration for the per-type QKV and output projections."""
        return UntiedLinearConfig(
            num_types=self.num_types,
            linear_projection=LinearProjectionConfig(
                input_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                bias=self.bias,
                init_weights=self.init_weights,
            ),
        )


class UntiedMLPConfig(BaseModel):
    """Configuration for an MLP with per-type (untied) weights.

    Composes an :class:`MLPConfig` (architecture: dims, activation, init) with a
    ``num_types`` field. The untied MLP mirrors :class:`MLP`'s topology
    (``input -> [hidden]*(num_layers+1) -> output`` with activations between layers)
    but uses :class:`UntiedLinear` for every linear layer.
    """

    num_types: int = Field(..., ge=1)
    """Number of distinct token types."""

    mlp: MLPConfig
    """Underlying MLP architecture (dims, activation, init)."""


class UntiedTransformerBlockConfig(BaseModel):
    """Configuration for a transformer block with per-type (untied) attention and MLP weights.

    Composes a :class:`TransformerBlockConfig` (shared layout: dims, heads, layer
    scale, drop path, etc.) with a ``num_types`` field. Both sub-layers have
    per-type weights: :class:`UntiedMultiHeadAttention` for attention and
    :class:`UntiedMLP` for the feed-forward.
    """

    num_types: int = Field(..., ge=1)
    """Number of distinct token types for the untied MLP."""

    transformer_block: TransformerBlockConfig
    """Shared transformer-block layout (dims, heads, layer scale, drop path, etc.)."""

    @computed_field
    def attention_config(self) -> UntiedMixedAttentionConfig:
        """Configuration for the UntiedMultiHeadAttention sub-layer."""
        block = self.transformer_block
        return UntiedMixedAttentionConfig(
            **block.model_dump(), **(block.attention_arguments or {}), num_types=self.num_types
        )

    @computed_field
    def untied_mlp_config(self) -> UntiedMLPConfig:
        """Configuration for the UntiedMLP sub-layer."""
        block = self.transformer_block
        # `mlp_hidden_dim` is guaranteed non-None by TransformerBlockConfig's validator.
        assert block.mlp_hidden_dim is not None
        return UntiedMLPConfig(
            num_types=self.num_types,
            mlp=MLPConfig(
                input_dim=block.hidden_dim,
                output_dim=block.hidden_dim,
                hidden_dim=block.mlp_hidden_dim,
                num_layers=0,
                activation="GELU",
                init_weights=block.init_weights,
            ),
        )


class UntiedPerceiverBlockConfig(BaseModel):
    """Configuration for a perceiver block with per-type (untied) Q/output projections and MLP weights.

    Composes a :class:`PerceiverBlockConfig` (shared layout: dims, heads, layer
    scale, drop path, etc.) with a ``num_types`` field. The Q and output
    projections in :class:`PerceiverAttention` become per-type via
    :class:`UntiedLinear`, while the KV projection stays shared (it operates on
    a single geometry encoding). The MLP is also replaced with
    :class:`UntiedMLP`.
    """

    num_types: int = Field(..., ge=1)
    """Number of distinct token types for the untied projections."""

    perceiver_block_config: PerceiverBlockConfig
    """Shared perceiver-block layout (dims, heads, kv_dim, layer scale, drop path, etc.)."""

    @computed_field
    def untied_mlp_config(self) -> UntiedMLPConfig:
        """Configuration for the UntiedMLP sub-layer."""
        block = self.perceiver_block_config
        assert block.mlp_hidden_dim is not None
        return UntiedMLPConfig(
            num_types=self.num_types,
            mlp=MLPConfig(
                input_dim=block.hidden_dim,
                output_dim=block.hidden_dim,
                hidden_dim=block.mlp_hidden_dim,
                num_layers=0,
                activation="GELU",
                init_weights=block.init_weights,
            ),
        )
