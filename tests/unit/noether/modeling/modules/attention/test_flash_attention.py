# Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from noether.modeling.modules.attention._flash_attention import (
    set_attn_impl,
    get_attn_impl,
    flash_attn_func,
    _sdpa_fallback,
)

def _detect_deps():
    cuda = torch.cuda.is_available()

    flash_attn = False
    kernels = False

    if cuda:
        try:
            import flash_attn_interface  # noqa
            flash_attn = True
        except ImportError:
            pass

        try:
            major, _ = torch.cuda.get_device_capability()
            if major >= 9:
                import kernels  # noqa
                kernels = True
        except Exception:
            pass

    return {
        "sdpa": True,
        "flash_attn_interface": flash_attn,
        "kernels": kernels,
    }
DEPS = _detect_deps()

def require(dep: str):
    if not DEPS.get(dep, False):
        pytest.skip(f"Missing dependency: {dep}")


@pytest.fixture(autouse=True)
def reset_backend(monkeypatch):
    monkeypatch.setenv("NOETHER_ATTN_IMPLEMENTATION", "")
    set_attn_impl(None)


ATTN_BACKEND_CASES = [
    ("sdpa", "sdpa", "sdpa"),
    ("invalid_value", "sdpa", "sdpa"),
    ("flash_attention_3", "flash_attention_3", "flash_attn_interface"),
    ("kernels-community/flash-attn3", "kernels-community/flash-attn3", "kernels"),
]
@pytest.fixture(params=ATTN_BACKEND_CASES)
def backend_case(request):
    return request.param

SHAPE_CASES = [
    (1, 8, 2, 16),
    (2, 16, 4, 32),
    (4, 32, 8, 64),
]
@pytest.fixture(params=SHAPE_CASES)
def shape_case(request):
    return request.param

def make_qkv(B, T, H, D, device):
    return (
        torch.randn(B, T, H, D, device=device),
        torch.randn(B, T, H, D, device=device),
        torch.randn(B, T, H, D, device=device),
    )

class TestBackendConfig:
    def test_env(self, monkeypatch, backend_case):
        env, expected, dep = backend_case
        require(dep)

        monkeypatch.setenv("NOETHER_ATTN_IMPLEMENTATION", env)
        set_attn_impl(None)

        assert get_attn_impl() == expected

    def test_override(self, backend_case):
        env, expected, dep = backend_case
        require(dep)

        set_attn_impl(env)
        assert get_attn_impl() == expected

class TestFlashAttentionFunctional:
    def test_forward_shape(self, backend_case, shape_case, device):
        env, _, dep = backend_case
        B, T, H, D = shape_case

        require(dep)
        set_attn_impl(env)

        q, k, v = make_qkv(B, T, H, D, device)
        out = flash_attn_func(q, k, v)

        assert out.shape == (B, T, H, D)

    def test_backward(self, backend_case, shape_case, device):
        env, _, dep = backend_case
        B, T, H, D = shape_case

        require(dep)
        set_attn_impl(env)

        q, k, v = make_qkv(B, T, H, D, device)

        q.requires_grad_()
        k.requires_grad_()
        v.requires_grad_()

        out = flash_attn_func(q, k, v)
        out.sum().backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

class TestSDPA:
    def test_shape(self, shape_case, device):
        B, T, H, D = shape_case
        q, k, v = make_qkv(B, T, H, D, device)

        out = _sdpa_fallback(q, k, v)
        assert out.shape == (B, T, H, D)

class TestCausalBehavior:
    def test_causal_invariance(self, backend_case, shape_case, device):
        env, _, dep = backend_case
        B, T, H, D = shape_case

        require(dep)
        set_attn_impl(env)

        q, k, v = make_qkv(B, T, H, D, device)
        out1 = flash_attn_func(q, k, v, causal=True)

        k2 = k.clone()
        k2[:, -1] += 100.0
        out2 = flash_attn_func(q, k2, v, causal=True)

        assert torch.allclose(out1[:, :T-1], out2[:, :T-1], atol=1e-5)

class TestParity:
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required",
    )
    def test_flash_vs_sdpa(self, shape_case, device):
        require("flash_attn_interface")
        B, T, H, D = shape_case

        set_attn_impl("flash_attention_3")

        q, k, v = make_qkv(B, T, H, D, device)

        out_flash = flash_attn_func(q, k, v)
        out_sdpa = F.scaled_dot_product_attention(
            q.transpose(1, 2), 
            k.transpose(1, 2), 
            v.transpose(1, 2)
        ).transpose(1, 2)

        assert torch.allclose(out_flash, out_sdpa, atol=1e-3, rtol=1e-3)