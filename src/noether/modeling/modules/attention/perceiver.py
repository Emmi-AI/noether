#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import einops
import torch
import torch.nn.functional as F
from pydantic import Field, model_validator
from torch import nn

from noether.core.schemas.modules.attention import AttentionConfig
from noether.modeling.functional.init import apply_init_method
from noether.modeling.functional.rope import rope


class PerceiverAttentionConfig(AttentionConfig):
    """Configuration for the Perceiver attention module."""

    kv_dim: int | None = Field(None)
    """Dimensionality of the key/value features. If None, use hidden_dim."""

    @model_validator(mode="after")
    def set_kv_dim(self):
        if self.kv_dim is None:
            self.kv_dim = self.hidden_dim
        return self


class PerceiverAttention(nn.Module):
    """Perceiver style attention module. This module is similar to a cross-attention module.

    Supports KV caching: when ``kv_cache`` is provided, the projected K/V tensors
    (with RoPE already applied) are loaded from the cache instead of being recomputed
    from ``kv``.
    """

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

        self.k = nn.Linear(config.kv_dim, config.hidden_dim, bias=config.bias)  # type: ignore[arg-type]
        self.v = nn.Linear(config.kv_dim, config.hidden_dim, bias=config.bias)  # type: ignore[arg-type]
        self.q = nn.Linear(config.hidden_dim, config.hidden_dim, bias=config.bias)
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=config.bias)
        self.dropout = config.dropout
        self.proj_dropout = nn.Dropout(config.dropout)
        if config.qk_norm:
            self.q_norm: nn.Module = nn.RMSNorm(self.head_dim)
            self.k_norm: nn.Module = nn.RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        apply_init_method(self, self.proj.weight, self.init_weights)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        q_freqs: torch.Tensor | None = None,
        k_freqs: torch.Tensor | None = None,
        kv_cache: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Forward function of the PerceiverAttention module.

        Args:
            q: Query tensor, shape (batch size, number of points/tokens, hidden_dim).
            kv: Key/value tensor, shape (batch size, number of latent tokens, kv_dim).
                Can be ``None`` when ``kv_cache`` is provided.
            attn_mask: When applying causal attention, an attention mask is required. Defaults to None.
            q_freqs: Frequencies for Rotary Positional Embedding (RoPE) of queries. None if use_rope=False.
            k_freqs: Frequencies for Rotary Positional Embedding (RoPE) of keys. None if use_rope=False.
                Not needed when loading from ``kv_cache`` (RoPE was already applied).
            kv_cache: Cached K/V tensors from a previous forward pass. Structure:
                ``{"k": tensor, "v": tensor}``.
                When provided, ``kv`` and ``k_freqs`` are ignored.

        Returns:
            Tuple of (output, new_kv_cache).
        """
        # Project query
        q = self.q(q)
        q = self.q_norm(
            einops.rearrange(
                q,
                "bs seqlen_q (num_heads head_dim) -> bs num_heads seqlen_q head_dim",
                num_heads=self.num_heads,
                head_dim=self.head_dim,
            )
        )

        if kv_cache is not None:
            # Load K/V from cache (already projected and RoPE-applied)
            k = kv_cache["k"]
            v = kv_cache["v"]
            new_cache = None
        else:
            # Project K/V from input
            if kv is None:
                raise ValueError("Either kv or kv_cache must be provided.")
            kv_weight = torch.cat([self.k.weight, self.v.weight], dim=0)
            kv_bias = torch.cat([self.k.bias, self.v.bias], dim=0) if self.k.bias is not None else None
            kv_proj = F.linear(kv, kv_weight, kv_bias)

            k, v = einops.rearrange(
                kv_proj,
                "bs seqlen_kv (two num_heads head_dim) -> two bs num_heads seqlen_kv head_dim",
                two=2,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
            ).unbind(0)
            k = self.k_norm(k)

            if self.use_rope:
                assert k_freqs is not None
                k = rope(k, freqs=k_freqs)

            new_cache = {"k": k, "v": v}

        # Apply RoPE to query
        if self.use_rope:
            assert q_freqs is not None
            q = rope(q, freqs=q_freqs)

        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        return self.proj_dropout(self.proj(x)), new_cache
