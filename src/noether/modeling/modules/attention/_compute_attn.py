#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Optional, Literal
import logging
import torch
import torch.nn.functional as F
from noether.core.schemas.modules.attention import ATTN_IMPLEMENTATION_REGISTRY
from noether.modeling.modules.attention._flash_attention import (
    get_attn_impl, 
    set_attn_impl, 
    flash_attn_func
)

logger = logging.getLogger(__name__)

def compute_attn_from_impl(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        dropout_p: float = 0.0,
        attn_implementation: str = "sdpa",
    ) -> torch.Tensor:
        """
        Compute attention using the configured attention implementation.
        
        Args:
            q: Query tensor of shape (batch_size, num_heads, seq_len_q, head_dim)
            k: Key tensor of shape (batch_size, num_heads, seq_len_k, head_dim)
            v: Value tensor of shape (batch_size, num_heads, seq_len_v, head_dim)
            attn_mask: Attention mask of shape (seq_len_q, seq_len_k)
            is_causal: Whether to apply causal masking (for autoregressive and flash attention)
            dropout_p: Dropout probability
        
        Returns:
            output: Tensor of shape (batch_size, num_heads, seq_len_q, head_dim)
        """
        if (attn_implementation not in ATTN_IMPLEMENTATION_REGISTRY) and (not "/" in attn_implementation):
            raise ValueError(f"Invalid attention implementation '{attn_implementation}'. Valid options are: {ATTN_IMPLEMENTATION_REGISTRY}")
        if get_attn_impl() != attn_implementation:
            set_attn_impl(attn_implementation)
            if attn_implementation != get_attn_impl():
                logger.warning(f"Attention implementation switched to {attn_implementation!r}.")
                attn_implementation = get_attn_impl()

        if attn_mask is not None and attn_implementation != "sdpa":
            # Flash attention does not support attn_mask, fallback to SDPA
            logger.warning("Using SDPA as fallback for attention mask.")
        if attn_implementation == "sdpa" or attn_mask is not None:
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
            )
            return attn_out # (batch_size, num_heads, seq_len_q, head_dim)
        else:
            assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4, "Flash attention requires q, k, v to be 4D tensors."
            # TODO: change the interface to accept shape (batch_size, seq_len, num_heads, head_dim) for q, k, v -> Would probably run 
            q = q.transpose(1, 2)  # (batch_size, seq_len_q, num_heads, head_dim)
            k = k.transpose(1, 2)  # (batch_size, seq_len_k, num_heads, head_dim)
            v = v.transpose(1, 2)  # (batch_size, seq_len_v, num_heads, head_dim)
            # Use Flash Attention (or other implementations)
            return flash_attn_func(q, k, v, dropout_p=dropout_p, causal=is_causal).transpose(1, 2)  # (batch_size, num_heads, seq_len_q, head_dim)
