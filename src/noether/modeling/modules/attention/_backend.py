#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Optional, Literal
import logging
import torch
import torch.nn.functional as F
from noether.core.schemas.modules.attention import AttnImplementation, ATTN_IMPLEMENTATION_REGISTRY
from noether.modeling.modules.attention._flash_attention import (
    get_attn_impl, 
    set_attn_impl, 
    flash_attn_func,
    flash_attn_qkvpacked_func,
    flash_attn_with_kvcache
)

logger = logging.getLogger(__name__)

class _AttentionKernel:
    def __init__(self, attn_implementation: AttnImplementation = "sdpa"):
        if attn_implementation not in ATTN_IMPLEMENTATION_REGISTRY:
            raise ValueError(f"Invalid attention implementation '{attn_implementation}'. Valid options are: {ATTN_IMPLEMENTATION_REGISTRY}")
        self.attn_implementation: AttnImplementation = attn_implementation
        if get_attn_impl() != attn_implementation:
            set_attn_impl(attn_implementation)

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        dropout_p: float = 0.0,    
    ) -> torch.Tensor:
        """
        Compute attention using the configured attention implementation.
        
        Args:
            q: Query tensor of shape (batch_size, seq_len_q, num_heads, head_dim)
            k: Key tensor of shape (batch_size, seq_len_k, num_heads, head_dim)
            v: Value tensor of shape (batch_size, seq_len_v, num_heads, head_dim)
            attn_mask: Attention mask of shape (seq_len_q, seq_len_k)
            is_causal: Whether to apply causal masking (for autoregressive and flash attention)
            dropout_p: Dropout probability
        
        Returns:
            output: Tensor of shape (batch_size, seq_len_q, num_heads, head_dim)
        """
        if attn_mask is not None and self.attn_implementation != "sdpa":
            # Flash attention does not support attn_mask, fallback to SDPA
            logger.warning("Using SDPA as fallback for attention mask.")
            
        if self.attn_implementation == "sdpa" or attn_mask is not None:
            q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len_q, head_dim)
            k = k.transpose(1, 2)  # (batch_size, num_heads, seq_len_k, head_dim)
            v = v.transpose(1, 2)  # (batch_size, num_heads, seq_len_v, head_dim)
            
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
            )
            return attn_out.transpose(1, 2)  # (batch_size, seq_len_q, num_heads, head_dim)
        else:
            # Use Flash Attention (or other implementations)
            return flash_attn_func(q, k, v, dropout_p=dropout_p, causal=is_causal)
        
def compute_attn_from_impl(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        dropout_p: float = 0.0,
        attn_implementation: AttnImplementation = "sdpa",
    ) -> torch.Tensor:
    """Compute attention using the specified attention implementation."""
    kernel = _AttentionKernel(attn_implementation)
    return kernel(q, k, v, attn_mask=attn_mask, is_causal=is_causal, dropout_p=dropout_p)
