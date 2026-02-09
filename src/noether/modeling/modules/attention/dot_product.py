#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import einops
import torch
import torch.nn.functional as F
from torch import nn

from noether.core.schemas.modules import AttentionConfig, DotProductAttentionConfig
from noether.modeling.functional.init import apply_init_method
from noether.modeling.functional.rope import rope


class DotProductAttention(nn.Module):
    """Scaled dot-product attention module."""

    def __init__(
        self,
        config: AttentionConfig,
    ):
        """

        Args:
            config: Configuration for the DotProductAttention module. See
                :class:`~noether.core.schemas.modules.AttentionConfig` for available options.
        """

        super().__init__()

        config = DotProductAttentionConfig(**config.model_dump())

        if not (config.hidden_dim % config.num_heads == 0):
            raise ValueError("The 'dim' must be divisible by 'num_heads'.")

        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.init_weights = config.init_weights
        self.use_rope = config.use_rope
        self.dropout = config.dropout
        self.proj_dropout = nn.Dropout(config.dropout)

        self.qkv = nn.Linear(config.hidden_dim, config.hidden_dim * 3, bias=config.bias)
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=config.bias)
        apply_init_method(self, self.proj.weight, self.init_weights)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        freqs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward function of the DotProductAttention module.

        Args:
            x: Tensor to apply self-attention over, shape (batch size, sequence length, hidden_dim).
            attn_mask: For causal attention (i.e., no attention over the future token) a attention mask should be provided. Defaults to None.
            freqs: Frequencies for Rotary Positional Embedding (RoPE) of queries/keys. None if use_rope=False.

        Returns:
            Returns the output of the attention module.
        """

        q, k, v = einops.rearrange(
            self.qkv(x),
            "bs seqlen (three num_heads head_dim) -> three bs num_heads seqlen head_dim",
            three=3,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).unbind(0)

        if self.use_rope:
            assert freqs is not None
            q = rope(q, freqs=freqs)
            k = rope(k, freqs=freqs)
        else:
            assert freqs is None

        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        x = self.proj_dropout(self.proj(x))

        return x
