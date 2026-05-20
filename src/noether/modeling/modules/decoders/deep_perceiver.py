#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

"""Contains a PerceiverDecoder implementation."""

from typing import Annotated, Any

import torch
from pydantic import BaseModel, Field
from torch import nn

from noether.core.schemas.mixins import InjectSharedFieldFromParentMixin, Shared
from noether.modeling.modules.blocks import PerceiverBlock, PerceiverBlockConfig


class DeepPerceiverDecoderConfig(InjectSharedFieldFromParentMixin, BaseModel):
    """Configuration for the DeepPerceiverDecoder module."""

    perceiver_block_config: Annotated[PerceiverBlockConfig, Shared] = Field(...)
    """Configuration for the Perceiver blocks used in the decoder."""

    depth: int = Field(1, ge=1)
    """Number of deep perceiver decoder layers (i.e., depth of the network). Defaults to 1."""

    input_dim: int = Field(..., ge=1)
    """Input dimension for the query positions."""


class DeepPerceiverDecoder(nn.Module):
    """A deep Perceiver decoder module. Can be configured with different number of layers and hidden dimensions.
    However, it should be noted that this layer is not a full-fledged Perceiver, since it only has a cross-attention mechanism.
    """

    def __init__(
        self,
        config: DeepPerceiverDecoderConfig,
    ):
        """

        Args:
            config: Configuration for the DeepPerceiverDecoder module. See :class:`~noether.core.schemas.modules.decoders.DeepPerceiverDecoderConfig` for available options.
        """
        super().__init__()

        # create query coordinate embeddings

        self.blocks = nn.ModuleList(
            [PerceiverBlock(config=config.perceiver_block_config) for _ in range(config.depth)],
        )

    def forward(
        self,
        kv: torch.Tensor,
        queries: torch.Tensor,
        attn_kwargs: dict[str, Any] | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            kv: The key-value tensor (batch_size, num_latent_tokens, dim).
            queries: The query tensor (batch_size, num_output_queries, dim).
            attn_kwargs: Dict with arguments for the attention (such as the attention mask or rope frequencies). Defaults to None.
            condition: Optional conditioning tensor that can be used in the attention mechanism. This can be used to pass additional conditioning information, etc.

        Returns:
            The predictions as sparse tensor (batch_size * num_output_pos, num_out_values).
        """
        assert kv.ndim == 3  # batch_size, num_latent_tokens, dim
        assert queries.ndim == 3  # batch_size, num_output_queries, pos_dim

        # perceiver
        for block in self.blocks:
            queries, _ = block(q=queries, kv=kv, attn_kwargs=attn_kwargs, condition=condition)

        return queries
