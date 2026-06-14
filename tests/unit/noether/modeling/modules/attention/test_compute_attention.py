# Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from noether.modeling.modules.attention._compute_attn import (
    compute_attn_from_impl,
)
from noether.modeling.modules.attention._flash_attention import (
    set_attn_impl,
    get_attn_impl,
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

def make_qkv(B, H, T, D, device):
    q = torch.randn(B, H, T, D, device=device)
    k = torch.randn(B, H, T, D, device=device)
    v = torch.randn(B, H, T, D, device=device)
    return q, k, v

@pytest.fixture(autouse=True)
def reset_backend(monkeypatch):
    monkeypatch.setenv("NOETHER_ATTN_IMPLEMENTATION", "")
    set_attn_impl(None)


SHAPE_CASES = [
    (1, 8, 2, 16),
    (2, 16, 4, 32),
    (4, 32, 8, 64),
]
@pytest.fixture(params=SHAPE_CASES)
def shape_case(request):
    return request.param

class TestComputeAttnDispatch:
    def test_sdpa_dispatch(self, shape_case, device):
        B, T, H, D = shape_case
        q, k, v = make_qkv(B, H, T, D, device)

        out = compute_attn_from_impl(
            q, k, v,
            attn_implementation="sdpa"
        )
        ref = F.scaled_dot_product_attention(q, k, v)

        assert torch.allclose(out, ref, atol=1e-5)

    def test_flash_dispatch(self, shape_case, device):
        require("flash_attn_interface")
        B, T, H, D = shape_case
        q, k, v = make_qkv(B, H, T, D, device)

        out = compute_attn_from_impl(
            q, k, v,
            attn_implementation="flash_attention_3"
        )

        ref = _sdpa_fallback(
            q.transpose(1, 2), 
            k.transpose(1, 2), 
            v.transpose(1, 2)
        ).transpose(1, 2)

        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)

class TestMaskBehavior:
    def test_mask_forces_sdpa(self, shape_case, device):
        B, T, H, D = shape_case
        q, k, v = make_qkv(B, H, T, D, device)

        attn_mask = torch.tril(torch.ones(T, T, device=device)).bool()
        out = compute_attn_from_impl(
            q, k, v,
            attn_mask=attn_mask,
            attn_implementation="flash_attention_3",
        )
        ref = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=False,
        )
        assert torch.allclose(out, ref, atol=1e-5)

class TestBackendState:
    def test_set_attn_impl_called(self, shape_case, device):
        B, T, H, D = shape_case
        q, k, v = make_qkv(B, H, T, D, device)

        compute_attn_from_impl(
            q, k, v,
            attn_implementation="sdpa"
        )

        assert get_attn_impl() == "sdpa"

    def test_flash_switches_backend(self, shape_case, device):
        require("flash_attn_interface")
        B, T, H, D = shape_case
        q, k, v = make_qkv(B, H, T, D, device)

        compute_attn_from_impl(
            q, k, v,
            attn_implementation="flash_attention_3"
        )
        assert get_attn_impl() == "flash_attention_3"


class TestValidation:

    def test_invalid_backend_raises(self, shape_case, device):
        B, T, H, D = shape_case
        q, k, v = make_qkv(B, H, T, D, device)

        with pytest.raises(ValueError):
            compute_attn_from_impl(
                q, k, v,
                attn_implementation="not_a_real_backend"
            )


class TestShapeContract:

    def test_shape_preservation(self, shape_case, device):
        B, T, H, D = shape_case
        q, k, v = make_qkv(B, H, T, D, device)

        out = compute_attn_from_impl(
            q, k, v,
            attn_implementation="sdpa"
        )

        assert out.shape == q.shape