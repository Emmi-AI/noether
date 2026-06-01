#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch
import torch.nn.functional as F
import os
from typing import Optional, Union
from types import SimpleNamespace

import math

_flash_attn = None
if is_torch_cuda_available():
    if is_flash_attn3_available_from_kernel():
        try:
            major, _ = torch.cuda_get_device_capability()
            if major >= 9:
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
                import kernels
                _flash_attn = kernels.get_kernel('varunneal/flash-attention-3').flash_attn_interface
        except:
            pass

flash_attention_is_installed = _flash_attn is not None

def _sdpa_fallback(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        dropout_p: float = 0.0, softmax_scale: float | None = None, window_size: tuple[int, int] = (-1, -1),
        alibi_slopes: torch.Tensor | None = None, deterministic: bool = False, use_gqa: bool = False):
    """Function of the scaled dot-product attention fallback.

    Args:
        q: Tensor to apply self-attention over, shape (batch size, sequence length, hidden_dim).
        k: Tensor to apply self-attention over, shape (batch size, sequence length, hidden_dim).
        v: Tensor to apply self-attention over, shape (batch size, sequence length, hidden_dim).
        dropout_p: Dropout probability for the attention weights.
        softmax_scale: Scale factor for the softmax operation.
        window_size: Size of the attention window.
        alibi_slopes: Alibi slopes for the attention mechanism.
        deterministic: Whether to use deterministic operations.
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
                is_causal=True, scale=softmax_scale, enable_gqa=use_gqa
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


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                window_size=(-1, -1), alibi_slopes=None, deterministic=False) -> torch.Tensor:
    if flash_attention_is_installed:
        return _flash_attn.flash_attn_func(
            q, k,v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic
        )
    use_gqa = (k.size(-2) != q.size(-2)) # GQA if Hq != Hkv
    return _sdpa_fallback(
        q, k, v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        use_gqa=use_gqa
    )

def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False,
                          window_size=(-1, -1), alibi_slopes=None, deterministic=False) -> torch.Tensor:
    if flash_attention_is_installed:
        return _flash_attn.flash_attn_qkvpacked_func(
            qkv,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            # alibi_slopes=alibi_slopes,
            # deterministic=deterministic
        )
    assert alibi_slopes is None, "Alibi slopes are not supported when FlashAttention is not installed."

    q, k, v = qkv.unbind(dim=2)
    return _sdpa_fallback(
        q, k, v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        # deterministic=deterministic
    )

def flash_attn_with_kvcache(
    q: torch.Tensor, # (B, Tq, H, D)
    k_cache: torch.Tensor, # (Bc, Tc, Hkv, D)
    v_cache: torch.Tensor, # (Bc, Tc, Hkv, D)
    k: Optional[torch.Tensor] = None, # (B, Tk, Hkv, D)
    v: Optional[torch.Tensor] = None, # (B, Tk, Hkv, D)
    rotary_cos=None, # ignored. handled outside
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=True,
    window_size=(-1, -1),  # -1 means 'infinite' context window
    rotary_interleaved=True,
    alibi_slopes=None,
) -> torch.Tensor:
    if (k is not None) and (v is not None):
        assert q.device == k.device == v.device, "q, k and v are expected to be on the same device"
    assert q.device == k_cache.device == v_cache.device, "q, k_cache and v_cache are expected to be on the same device"
    if flash_attention_is_installed:
        return _flash_attn.flash_attn_with_kvcache(
            q, k_cache, v_cache,
            k=k,
            v=v,
            # rotary_cos=rotary_cos,
            # rotary_sin=rotary_sin,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=cache_batch_idx,
            # block_table=block_table,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            # rotary_interleaved=rotary_interleaved,
            # alibi_slopes=alibi_slopes
        )
    Tq = q.size(1)
    
    cur_pos = cache_seqlens[0].item() # TODO: change for batch support -> efficiently get max cur_pos
    end_pos = cur_pos + Tq

    if k is not None and v is not None:
        k_cache[:,cur_pos:end_pos,:,:] = k
        v_cache[:,cur_pos:end_pos,:,:] = v
    
    k = k_cache[:,:end_pos,:,:]
    v = v_cache[:,:end_pos,:,:]
    use_gqa = (k.size(-2) != q.size(-2)) # GQA if Hq != Hkv
    return _sdpa_fallback(
        q, k, v,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        window_size=window_size,
        enable_gqa=use_gqa
    )