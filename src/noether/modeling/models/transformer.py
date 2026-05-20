#  Copyright © 2025 Emmi AI GmbH. All rights reserved.


from typing import Annotated

import torch
import torch.nn as nn
from pydantic import ConfigDict, Field

from noether.core.models.base import ModelBaseConfig
from noether.core.schemas.mixins import InjectSharedFieldFromParentMixin, Shared
from noether.modeling.modules.blocks import TransformerBlock
from noether.modeling.modules.blocks.transformer import TransformerBlockConfig


class TransformerConfig(ModelBaseConfig, InjectSharedFieldFromParentMixin):
    """Configuration for a Transformer model."""

    model_config = ConfigDict(extra="forbid")

    hidden_dim: int = Field(..., ge=1)
    """Hidden dimension of the model. Used for all transformer blocks."""

    depth: int = Field(..., ge=1)
    """Number of transformer blocks in the model."""

    transformer_block_config: Annotated[TransformerBlockConfig, Shared]


class Transformer(nn.Module):
    """Implementation of a Transformer model."""

    def __init__(
        self,
        config: TransformerConfig,
    ):
        """
        Args:
            config: Configuration of the Transformer model.
        """
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config=config.transformer_block_config,  # type: ignore[arg-type]
                )
                for _ in range(config.depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_kwargs: dict[str, torch.Tensor],
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the Transformer model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim).
            attn_kwargs: Additional arguments for the attention mechanism.
            condition: Optional conditioning vector of shape (batch_size, condition_dim) consumed
                by each block's AdaLN-Zero modulation. ``None`` (default) for unconditioned models.
        Returns:
            torch.Tensor: Output tensor after processing through the Transformer model.
        """

        for block in self.blocks:
            x, _ = block(x, condition=condition, attn_kwargs=attn_kwargs)

        return x
