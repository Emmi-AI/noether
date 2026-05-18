#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from pydantic import ConfigDict, Field, computed_field

from noether.core.schemas.modules.blocks import TransformerBlockConfig

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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def transformer_block_config(self) -> TransformerBlockConfig:
        return TransformerBlockConfig(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            mlp_expansion_factor=self.mlp_ratio,
            attention_constructor="dot_product",
            condition_dim=self.hidden_dim if self.use_conditioning else None,
            use_rope=True,
            dropout=self.attn_drop,
            init_weights="xavier",
        )
