#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch
import torch.nn.functional as F
import os
from typing import Optional, Union, Literal

from noether.core.schemas.modules.attention import ATTN_IMPLEMENTATION_REGISTRY

import logging

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    try:
        import flash_attn_interface
        _is_flash_attn_3_available = True
    except ImportError:
        _is_flash_attn_3_available = False

    try:
        major, _ = torch.cuda.get_device_capability()
        if major >= 9:
            import kernels
            _is_kernels_available = True
        else:
            _is_kernels_available = False
    except ImportError:
        _is_kernels_available = False
else:
    _is_flash_attn_3_available = False
    _is_kernels_available = False


_attn_mode: Optional[str] = None
_flash_attn = None

def _init_attn_mode(override: Optional[str] = None):
    """Initialize _attn_mode and _flash_attn ONCE (thread-safe)."""
    global _attn_mode, _flash_attn
    if _attn_mode is not None and override is None:
        return  # Already initialized

    # Priority: override > environment > default
    mode = override or os.getenv("NOETHER_ATTN_IMPLEMENTATION", "sdpa").lower()
    valid_modes = ATTN_IMPLEMENTATION_REGISTRY

    if (mode not in valid_modes) and ("/" not in mode):
        if override:
            substr = "attention implementation 'override'"
        else:
            substr = "environment variable NOETHER_ATTN_IMPLEMENTATION"
        logger.warning(
            f"Invalid {substr}='{mode}'. Falling back to 'sdpa'. Valid options: {valid_modes}"
        )
        mode = "sdpa"
    _attn_mode = mode

    _flash_attn = None
    if mode == "sdpa":
        return
    if mode == "flash_attention_3":
        if not _is_flash_attn_3_available:
            logger.warning(
                "Flash Attention (flash_attention_3) is not available. Falling back to 'sdpa'."
            )
            _attn_mode = "sdpa"
            return
        try:
            import flash_attn_interface
            _flash_attn = flash_attn_interface
        except Exception as e:
            logger.warning(
                f"Failed to load Flash Attention (flash_attention_3): {str(e)}. Falling back to 'sdpa'."
            )
            _attn_mode = "sdpa"
    elif "/" in mode:
        if not _is_kernels_available:
            logger.warning(
                f"Custom kernel implementation '{mode}' is not available. Falling back to 'sdpa'."
            )
            _attn_mode = "sdpa"
            return
        try:
            major, _ = torch.cuda.get_device_capability()
            if major >= 9:
                import kernels
                _flash_attn = kernels.get_kernel(mode).flash_attn_interface
        except Exception as e:
            logger.warning(
                f"Failed to load Flash Attention kernel {mode!r}: {str(e)}. Falling back to 'sdpa'."
            )
            _attn_mode = "sdpa"
    else:
        logger.warning(
            f"Invalid attention implementation '{mode}'. Falling back to 'sdpa'. Valid options: {valid_modes} or a kernel path from `huggingface/kernels` named as '<org>/<kernel_name>' (eg: 'kernels-community/flash-attn3')."
        )
        

def set_attn_impl(impl: Optional[str] = None):
    """Programmatically override the attention implementation (for CLI/config)."""
    _init_attn_mode(override=impl)

def get_attn_impl() -> str:
    """Get the current attention implementation (cached)."""
    if _attn_mode is None:
        _init_attn_mode()
    return _attn_mode

def _is_flash_attention_installed() -> bool:
    """Check if Flash Attention is available (cached)."""
    if _attn_mode is None:
        _init_attn_mode()
    return _flash_attn is not None

def _sdpa_fallback(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False,
        dropout_p: float = 0.0, softmax_scale: float | None = None, window_size: tuple[int, int] = (-1, -1),
        use_gqa: bool = False):
    """Function of the scaled dot-product attention fallback with flash-attention signature.
    Supports causal and non-causal attention, with flash-attention-2 by default.

    Args:
        q: Tensor to apply self-attention over, shape (batch size, sequence length, hidden_dim).
        k: Tensor to apply self-attention over, shape (batch size, sequence length, hidden_dim).
        v: Tensor to apply self-attention over, shape (batch size, sequence length, hidden_dim).
        is_causal: Whether to apply causal masking (default: False).
        dropout_p: Dropout probability for the attention weights.
        softmax_scale: Scale factor for the softmax operation.
        window_size: Size of the attention window.
        use_gqa: Whether to use grouped query attention.

    Returns:
        Returns the output of the attention module.
    """
    Tq, Tk = q.size(1), k.size(1)
    if window_size is None:
        window_size = (-1, -1)
    window = window_size[0]
    q = q.transpose(1, 2)  # B, H, Tq, D
    k = k.transpose(1, 2)  # B, H, Tk, D
    v = v.transpose(1, 2)  # B, H, Tk, D
    
    device = q.device
    if Tq == 1:
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p,
            is_causal=False, scale=softmax_scale
        )
    elif Tq == Tk:
        if window < 0 or window >= Tq:
            output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout_p,
                is_causal=is_causal, scale=softmax_scale, enable_gqa=use_gqa
            )
        else:
            mask = torch.triu(torch.ones((Tq, Tq), device=device), diagonal=1)
            mask = torch.logical_or(mask, torch.tril(torch.ones((Tq, Tq), device=device), diagonal=-window-1))
            output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=dropout_p,
                is_causal=False, scale=softmax_scale, enable_gqa=use_gqa
            )
    else:
        prefix_len = Tk - Tq
        mask = torch.zeros((Tq, Tk), device=device)
        mask = torch.zeros((Tq, Tk), device=device, dtype=torch.bool)
        mask = mask.masked_fill(torch.tril(torch.ones((Tq, Tk), device=device), diagonal=-prefix_len-1) == 1, True)
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=dropout_p, 
            is_causal=False, scale=softmax_scale, enable_gqa=use_gqa
        )
    return output.transpose(1, 2)  # B, Tq, H, D


def flash_attn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
        dropout_p: float = 0.0, softmax_scale: float | None = None, 
        causal: bool = False, window_size: tuple[int, int] = (-1, -1)) -> torch.Tensor:
    
    if _is_flash_attention_installed():
        return _flash_attn.flash_attn_func(
            q, k,v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )
    use_gqa = (k.size(-2) != q.size(-2)) # GQA if Hq != Hkv
    return _sdpa_fallback(
        q, k, v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        window_size=window_size,
        use_gqa=use_gqa,
        is_causal=causal
    )
