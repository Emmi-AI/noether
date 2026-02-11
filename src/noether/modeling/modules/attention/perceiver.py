#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import einops
import torch
import torch.nn.functional as F
from torch import nn

from noether.core.schemas.modules import AttentionConfig, PerceiverAttentionConfig
from noether.modeling.functional.init import apply_init_method
from noether.modeling.functional.rope import rope


class PerceiverAttention(nn.Module):
    """Perceiver style attention module. This module is similar to a cross-attention modules."""

    def __init__(
        self,
        config: AttentionConfig,
    ):
        """

        Args:
            config: Configuration for the PerceiverAttention module. See
                :class:`~noether.core.schemas.modules.AttentionConfig` for available options.
        """

        super().__init__()

        config = PerceiverAttentionConfig(**config.model_dump())

        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.init_weights = config.init_weights
        self.use_rope = config.use_rope

        self.kv = nn.Linear(config.kv_dim, config.hidden_dim * 2, bias=config.bias)  # type: ignore[arg-type]
        self.q = nn.Linear(config.hidden_dim, config.hidden_dim, bias=config.bias)
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=config.bias)
        self.dropout = config.dropout
        self.proj_dropout = nn.Dropout(config.dropout)

        apply_init_method(self, self.proj.weight, self.init_weights)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        q_freqs: torch.Tensor | None = None,
        k_freqs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward function of the PerceiverAttention module.

        Args:
            q: Query tensor, shape (batch size, number of points/tokens, hidden_dim).
            kv: Key/value tensor, shape (batch size, number of latent tokens, hidden_dim).
            attn_mask: When applying causal attention, an attention mask is required. Defaults to None.
            q_freqs: Frequencies for Rotary Positional Embedding (RoPE) of queries. None if use_rope=False.
            k_freqs: Frequencies for Rotary Positional Embedding (RoPE) of keys. None if use_rope=False.

        Returns:
            Returns the output of the perceiver attention module.
        """
        # project to attention space
        kv = self.kv(kv)
        q = self.q(q)

        # split per head
        q = einops.rearrange(
            q,
            "bs seqlen_q (num_heads head_dim) -> bs num_heads seqlen_q head_dim",
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        k, v = einops.rearrange(
            kv,
            "bs seqlen_kv (two num_heads head_dim) -> two bs num_heads seqlen_kv head_dim",
            two=2,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).unbind(0)

        if self.use_rope:
            assert q_freqs is not None
            assert k_freqs is not None
            q = rope(q, freqs=q_freqs)
            k = rope(k, freqs=k_freqs)
        else:
            assert q_freqs is None
            assert k_freqs is None

        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        return self.proj_dropout(self.proj(x))  # type: ignore[no-any-return]
