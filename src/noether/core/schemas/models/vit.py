#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from pydantic import ConfigDict, Field

from .base import ModelBaseConfig


class ViTConfig(ModelBaseConfig):
    """Configuration for ViT model"""

    model_config = ConfigDict(extra="forbid")

    coord_dim: int = Field(..., ge=1)
    """Coordinate dimensionality of the input grid (2 for 2D, 3 for 3D)."""

    out_channels: int = Field(..., ge=1)
    """Number of output channels emitted per spatial cell."""

    patch_size: int = Field(..., ge=2)
    """Patch side length in cells. The grid resolution must be divisible by this value."""

    hidden_dim: int = Field(192, ge=1)
    """Token hidden dimension throughout the transformer stack."""

    num_heads: int = Field(6, ge=1)
    """Number of attention heads in each transformer block."""

    depth: int = Field(10, ge=1)
    """Number of stacked transformer blocks."""

    mlp_ratio: int = Field(4, ge=1)
    """FFN expansion factor inside each transformer block."""

    use_conditioning: bool = True
    """If True, enable AdaLN-Zero conditioning (forward requires ``cond``); if False, plain ViT (``cond`` must be ``None``)."""

    token_dropout: float = Field(0.0, ge=0.0, le=1.0)
    """Per-patch token dropout probability used during training."""

    attn_drop: float = Field(0.0, ge=0.0, le=1.0)
    """Dropout probability inside attention."""

    use_conv_output_head: bool = True
    """If True, decode via a cascaded PixelShuffle conv head; if False, decode via a linear unpatchify."""
