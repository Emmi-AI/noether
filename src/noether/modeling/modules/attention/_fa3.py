#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Optional FlashAttention-3 / FlexAttention routing for ``F.scaled_dot_product_attention``.

:func:`sdpa` is a drop-in replacement for ``torch.nn.functional.scaled_dot_product_attention``
with two opt-in fast paths:

* **FA3** (``NOETHER_USE_FA3=1``): for unmasked self/cross attention on CUDA
  bf16/fp16 inputs.
* **FlexAttention** (``NOETHER_USE_FLEX=1``): for KV-padding-masked attention
  whose ``attn_mask`` arrives in the broadcast-friendly ``(B, 1, 1, S_kv)``
  bool layout that :class:`MixedAttention` produces. This avoids materializing
  a dense attention matrix with the padding rows zeroed out, which is the
  current SDPA fallback.

Activation of either path is independent. When both flags are unset the call
collapses to a plain ``F.scaled_dot_product_attention`` so flipping flags is
risk-free.

FA3 eligibility (all required):

* FA3 importable (i.e. ``flash_attn_3`` was built and installed).
* Input is on CUDA in ``bf16`` or ``fp16``.
* No element-wise ``attn_mask`` (FA3 only supports ``causal`` / ``window_size``).
* ``dropout_p == 0.0`` (FA3's ``flash_attn_func`` does not accept dropout).

FlexAttention eligibility (all required):

* PyTorch ships ``torch.nn.attention.flex_attention`` (>=2.5).
* Input is on CUDA in ``bf16`` or ``fp16``.
* ``attn_mask`` is a bool tensor with shape ``(B, 1, 1, S_kv)`` — exactly the
  KV-padding broadcast emitted by :class:`MixedAttention`.
* ``dropout_p == 0.0`` and ``is_causal is False``.
"""

from __future__ import annotations

import logging
import os
from typing import cast

import torch
import torch.nn.functional as F

_log = logging.getLogger(__name__)

_FA3_FLAG_ENV = "NOETHER_USE_FA3"
_FLEX_FLAG_ENV = "NOETHER_USE_FLEX"

# FlexAttention's Triton kernel only safely lowers for power-of-two head_dims.
# Non-blessed head_dims (e.g. 48 = hidden_dim 576 / 12 heads) trigger
# ``LoweringException: NameError: name 'FloatTrueDiv' is not defined`` from
# Inductor when it tries to bake ``SM_SCALE = 1/sqrt(head_dim)`` as a kernel
# constant under symbolic shapes. Gate Flex on this set to avoid the crash.
_FLEX_BLESSED_HEAD_DIMS: frozenset[int] = frozenset({16, 32, 64, 128, 256})


def _resolve_fa3():
    """Resolve and cache ``flash_attn_3.flash_attn_func`` if available.

    Returns ``None`` when the env flag is unset or the package is missing,
    in which case :func:`sdpa` reduces to a plain SDPA call.
    """
    if os.getenv(_FA3_FLAG_ENV, "0") != "1":
        return None
    try:
        # The FA3 wheel ships ``flash_attn_interface`` at the top level of
        # site-packages and the compiled C++ extension at ``flash_attn_3._C``.
        # Importing the interface module triggers the C++ op registration as
        # a side-effect.
        from flash_attn_interface import flash_attn_func
    except ImportError:
        return None
    return flash_attn_func


def _resolve_flex():
    """Resolve ``flex_attention`` and ``create_block_mask`` if available.

    Returns ``(flex_attention, create_block_mask)`` or ``None``. The callable
    is **not** pre-wrapped in ``torch.compile``: Dynamo special-cases
    ``flex_attention`` as a higher-order op when it is called inside an
    outer compiled region (the project's training path, see commit e6a8308),
    and lowers it directly into the model's graph. Pre-compiling here
    creates a nested compile that forces Inductor to lower flex with
    symbolic shapes from the outer graph, triggering
    ``LoweringException: NameError: name 'FloatTrueDiv' is not defined`` for
    head_dims like 48 and composite sequence lengths like
    ``n_anchors + n_wells + n_queries``. ``dynamic=True`` makes this worse,
    not better — the bug is the nested compile, not the dynamic-shape mode.

    Calling uncompiled ``flex_attention`` from eager code emits a PyTorch
    warning that the backward pass may be incorrect; that warning is real
    but only applies when this dispatcher runs outside a compiled model
    (e.g. unit tests). Production runs are inside ``torch.compile``, so
    Dynamo handles compilation for us.
    """
    if os.getenv(_FLEX_FLAG_ENV, "0") != "1":
        return None
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    except ImportError:
        return None
    return flex_attention, create_block_mask


_FA3_FUNC = _resolve_fa3()
_FLEX = _resolve_flex()

if _FA3_FUNC is not None:
    _log.info("FlashAttention-3 enabled (NOETHER_USE_FA3=1)")
elif os.getenv(_FA3_FLAG_ENV, "0") == "1":
    _log.warning(
        "NOETHER_USE_FA3=1 but flash_attn_interface is not importable; "
        "falling back to torch SDPA. Install FA3 with: "
        "uv pip install --no-build-isolation third_party/flash-attention/hopper"
    )
else:
    _log.info("FlashAttention-3 disabled; using torch SDPA (set NOETHER_USE_FA3=1 to enable)")


def fa3_available() -> bool:
    """Whether the FA3 path is currently active and importable."""
    return _FA3_FUNC is not None


def flex_available() -> bool:
    """Whether the FlexAttention path is currently active and importable."""
    return _FLEX is not None


def _is_kv_padding_mask(attn_mask: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> bool:
    """Return True iff ``attn_mask`` is the ``(B, 1, 1, S_kv)`` bool broadcast.

    This is the canonical shape produced by :class:`MixedAttention` when a
    ``key_padding_mask`` is supplied — every query attends to the same set of
    valid keys, so ``mask[b, 0, 0, kv]`` fully describes the pattern.
    """
    return (
        attn_mask.dtype == torch.bool
        and attn_mask.ndim == 4
        and attn_mask.shape[0] == q.shape[0]
        and attn_mask.shape[1] == 1
        and attn_mask.shape[2] == 1
        and attn_mask.shape[3] == k.shape[2]
    )


def _flex_attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    """Run FlexAttention with a per-batch KV-padding mask.

    ``attn_mask`` must be ``(B, 1, 1, S_kv)`` bool; this is converted to a
    ``BlockMask`` so flex skips fully-padded blocks instead of softmaxing
    over them.
    """
    assert _FLEX is not None
    flex_fn, create_block_mask = _FLEX

    # ``reshape`` (not ``view``) — the canonical mask layout is built via
    # ``kv_bool[:, None, None, :]`` in MixedAttention, which is non-contiguous;
    # ``view`` would raise a stride error here.
    mask_2d = attn_mask.reshape(attn_mask.shape[0], attn_mask.shape[3]).contiguous()  # (B, S_kv)

    def mask_mod(b, _h, _q_idx, kv_idx):
        return mask_2d[b, kv_idx]

    block_mask = create_block_mask(
        mask_mod,
        B=q.shape[0],
        H=None,
        Q_LEN=q.shape[2],
        KV_LEN=k.shape[2],
        device=q.device.type,
    )
    return cast("torch.Tensor", flex_fn(q, k, v, block_mask=block_mask))


def sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """Drop-in :func:`F.scaled_dot_product_attention` with optional FA3 / Flex routing.

    Inputs follow the BHND layout ``(batch, num_heads, seqlen, head_dim)`` used
    by PyTorch SDPA. FA3 expects BSHD; the transpose around the FA3 call is
    a metadata-only view (no copy) for contiguous inputs. FlexAttention shares
    the BHND layout so no transpose is needed there.

    Dispatch order: FA3 (no mask) → FlexAttention (KV-padding mask) → vanilla SDPA.

    Args:
        q: Query tensor ``(B, H, S_q, D)``.
        k: Key tensor ``(B, H_kv, S_k, D)``. ``H_kv`` may differ from ``H``
            for MQA/GQA; FA3 supports the divisibility case.
        v: Value tensor ``(B, H_kv, S_k, D)``.
        attn_mask: Additive or boolean attention mask. The FA3 path is skipped
            when this is supplied (FA3 only supports causal / sliding-window
            via the ``causal`` / ``window_size`` arguments). The FlexAttention
            path activates only for the KV-padding broadcast layout
            ``(B, 1, 1, S_kv)`` with bool dtype.
        dropout_p: Dropout probability. FA3 / Flex are bypassed when nonzero.
        is_causal: Whether to apply a causal mask. Forwarded to FA3 as
            ``causal=True``; Flex path is bypassed when this is set.

    Returns:
        Attention output ``(B, H, S_q, D)`` matching the PyTorch SDPA layout.
    """
    if (
        _FA3_FUNC is not None
        and attn_mask is None
        and dropout_p == 0.0
        and q.is_cuda
        and q.dtype in (torch.bfloat16, torch.float16)
    ):
        # BHND -> BSHD (no copy if contiguous).
        q3 = q.transpose(1, 2)
        k3 = k.transpose(1, 2)
        v3 = v.transpose(1, 2)
        out = _FA3_FUNC(q3, k3, v3, causal=is_causal)
        # FA3 returns ``(out, lse)`` only when ``return_attn_probs=True``;
        # with the default it returns ``out`` directly.
        if isinstance(out, tuple):
            out = out[0]
        # BSHD -> BHND.
        return cast("torch.Tensor", out.transpose(1, 2))

    if (
        _FLEX is not None
        and attn_mask is not None
        and dropout_p == 0.0
        and not is_causal
        and q.is_cuda
        and q.dtype in (torch.bfloat16, torch.float16)
        # and q.shape[-1] in _FLEX_BLESSED_HEAD_DIMS  # temporarily disabled
        and _is_kv_padding_mask(attn_mask, q, k)
    ):
        return _flex_attend(q, k, v, attn_mask)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
