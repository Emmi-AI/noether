#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Parity tests for the :mod:`noether.modeling.modules.attention._fa3` dispatcher.

The vanilla SDPA fallback is the reference. Each opt-in path (FA3, FlexAttention)
must agree with it within numerical tolerance for the input regime it accepts.
"""

from __future__ import annotations

import importlib

import pytest
import torch
import torch.nn.functional as F

from noether.modeling.modules.attention import _fa3

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def _reload_fa3(monkeypatch: pytest.MonkeyPatch, *, fa3: bool, flex: bool) -> None:
    """Toggle the env flags and reload the module so cached resolvers re-run."""
    monkeypatch.setenv(_fa3._FA3_FLAG_ENV, "1" if fa3 else "0")
    monkeypatch.setenv(_fa3._FLEX_FLAG_ENV, "1" if flex else "0")
    importlib.reload(_fa3)


def _make_qkv(B: int, H: int, S: int, D: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    return q, k, v


@requires_cuda
def test_dispatch_unmasked_no_flags_matches_sdpa(monkeypatch) -> None:
    """With both flags off the dispatcher must equal F.scaled_dot_product_attention exactly."""
    _reload_fa3(monkeypatch, fa3=False, flex=False)
    q, k, v = _make_qkv(2, 4, 256, 64)
    out_ref = F.scaled_dot_product_attention(q, k, v)
    out_new = _fa3.sdpa(q, k, v)
    assert torch.equal(out_ref, out_new), "Dispatcher with no flags must be a literal SDPA call"


def test_flex_unavailable_when_flag_off(monkeypatch) -> None:
    _reload_fa3(monkeypatch, fa3=False, flex=False)
    assert not _fa3.flex_available()


@requires_cuda
def test_flex_path_matches_sdpa_with_kv_padding(monkeypatch) -> None:
    """FlexAttention output ≈ SDPA with the same broadcast bool mask."""
    _reload_fa3(monkeypatch, fa3=False, flex=True)
    if not _fa3.flex_available():
        pytest.skip("flex_attention not available in this torch build")

    B, H, S, D = 2, 4, 256, 64
    q, k, v = _make_qkv(B, H, S, D)

    # Half the keys are padded in batch 0; batch 1 is fully real. Mirrors the
    # MixedAttention layout when wells.mask has padded entries.
    kv_keep = torch.ones(B, S, dtype=torch.bool, device="cuda")
    kv_keep[0, S // 2 :] = False
    attn_mask = kv_keep[:, None, None, :]

    out_ref = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    out_new = _fa3.sdpa(q, k, v, attn_mask=attn_mask)

    diff = (out_ref.float() - out_new.float()).abs().max().item()
    assert diff < 1e-2, f"flex/sdpa divergence {diff:.3e} exceeds bf16 tolerance"


@requires_cuda
def test_flex_path_skipped_for_non_kv_padding_mask(monkeypatch) -> None:
    """A non-broadcast mask shape must fall back to vanilla SDPA, not crash flex."""
    _reload_fa3(monkeypatch, fa3=False, flex=True)
    if not _fa3.flex_available():
        pytest.skip("flex_attention not available in this torch build")

    B, H, S, D = 2, 4, 64, 32
    q, k, v = _make_qkv(B, H, S, D)
    # Per-(query, key) mask — flex path is gated off, dispatcher must reach SDPA.
    attn_mask = torch.ones(B, 1, S, S, dtype=torch.bool, device="cuda")
    attn_mask[0, 0, :, S // 2 :] = False

    out_ref = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    out_new = _fa3.sdpa(q, k, v, attn_mask=attn_mask)
    assert torch.equal(out_ref, out_new)


@requires_cuda
def test_flex_path_skipped_for_unblessed_head_dim(monkeypatch) -> None:
    """head_dim ∉ {16, 32, 64, 128, 256} must fall back to SDPA-with-mask.

    Non-blessed head_dims (e.g. 48) trigger an Inductor LoweringException when
    flex_attention is lowered with ``SM_SCALE = 1/sqrt(head_dim)`` as a
    symbolic constant. The dispatcher must skip Flex for these cases.
    """
    _reload_fa3(monkeypatch, fa3=False, flex=True)
    if not _fa3.flex_available():
        pytest.skip("flex_attention not available in this torch build")

    B, H, S, D = 2, 4, 64, 48  # head_dim=48 is non-blessed
    q, k, v = _make_qkv(B, H, S, D)
    kv_keep = torch.ones(B, S, dtype=torch.bool, device="cuda")
    attn_mask = kv_keep[:, None, None, :]

    out_ref = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    out_new = _fa3.sdpa(q, k, v, attn_mask=attn_mask)
    assert torch.equal(out_ref, out_new), "head_dim=48 must fall through to SDPA, not flex"


@requires_cuda
def test_flex_path_skipped_when_causal(monkeypatch) -> None:
    """``is_causal=True`` forces the vanilla SDPA path even with flex enabled."""
    _reload_fa3(monkeypatch, fa3=False, flex=True)
    if not _fa3.flex_available():
        pytest.skip("flex_attention not available in this torch build")

    B, H, S, D = 2, 4, 64, 32
    q, k, v = _make_qkv(B, H, S, D)
    kv_keep = torch.ones(B, S, dtype=torch.bool, device="cuda")
    attn_mask = kv_keep[:, None, None, :]

    # Dispatcher with is_causal=True must reproduce SDPA(mask + causal) exactly,
    # not silently swap in flex (which doesn't combine our mask with a causal mask).
    out_ref = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=True)
    out_new = _fa3.sdpa(q, k, v, attn_mask=attn_mask, is_causal=True)
    assert torch.equal(out_ref, out_new)
