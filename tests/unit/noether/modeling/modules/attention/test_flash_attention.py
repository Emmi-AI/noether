# Copyright © 2025 Emmi AI GmbH. All rights reserved.
import pytest
import torch
from noether.core.schemas.modules.attention import AttentionConfig
from noether.modeling.modules.attention._flash_attention import set_attn_impl
from noether.modeling.modules.attention.dot_product import DotProductAttention, DotProductAttentionConfig
from .test_dot_product import (  # Reuse existing test functions
    attention_module,
    test_eval_mode,
    test_forward_shape,
    test_forward_with_mask,
    test_no_bias,
    test_truncnormal_init0,
    test_invalid_dim_num_heads,
    test_reset_parameters_invalid,
)

# --- GPU Tests (Reuse CPU test logic) ---
@pytest.mark.gpu  # Matches your existing integration marker
class TestDotProductAttentionGPU:
    """GPU tests for DotProductAttention with all backends."""

    @pytest.fixture
    def attention_module(self, gpu_device, attn_impl):
        """GPU fixture for attention module, parametrized over backends."""
        torch.manual_seed(42)
        config = DotProductAttentionConfig(
            hidden_dim=16,
            num_heads=4,
            init_weights="truncnormal002",
            attn_impl=attn_impl if attn_impl != "sdpa" else None,
        )
        return DotProductAttention(config).to(device=gpu_device)

    def test_eval_mode_gpu(self, gpu_device, attn_impl):
        """Test eval mode on GPU."""
        config = AttentionConfig(
            hidden_dim=16,
            num_heads=4,
            init_weights="truncnormal002",
            attn_impl=attn_impl if attn_impl != "sdpa" else None,
        )
        model = DotProductAttention(config).to(device=gpu_device)
        model.eval()
        assert not model.training, "Model should be in eval mode"

    def test_forward_shape_gpu(self, attention_module, gpu_device):
        """Test forward shape on GPU (reuses CPU test logic)."""
        torch.manual_seed(42)
        x = torch.randn(2, 10, 16, device=gpu_device)
        output = attention_module(x)
        assert output.shape == (2, 10, 16), "Output shape mismatch"
        assert output.device.type == "cuda", "Output should be on GPU"

    def test_forward_with_mask_gpu(self, attention_module, gpu_device):
        """Test forward with mask on GPU."""
        torch.manual_seed(42)
        x = torch.randn(2, 10, 16, device=gpu_device)
        attn_mask = torch.zeros(10, 10, device=gpu_device)
        output = attention_module(x, attn_mask=attn_mask)
        assert output.shape == (2, 10, 16), "Output shape mismatch with attention mask"
        assert output.device.type == "cuda", "Output should be on GPU"

    def test_is_causal_gpu(self, attention_module, gpu_device):
        """Test is_causal flag on GPU."""
        torch.manual_seed(42)
        x = torch.randn(2, 10, 16, device=gpu_device)
        output = attention_module(x, is_causal=True)
        assert output.shape == (2, 10, 16), "Output shape mismatch with causal attention"
        assert output.device.type == "cuda", "Output should be on GPU"

    def test_no_bias_gpu(self, gpu_device, attn_impl):
        """Test no bias on GPU."""
        config = DotProductAttentionConfig(
            hidden_dim=4,
            num_heads=2,
            bias=False,
            attn_impl=attn_impl if attn_impl != "sdpa" else None,
        )
        attn = DotProductAttention(config).to(device=gpu_device)
        assert attn.q.bias is None
        assert attn.k.bias is None
        assert attn.v.bias is None
        assert attn.proj.bias is None

    def test_truncnormal_init0_gpu(self, gpu_device, attn_impl):
        """Test truncnormal init on GPU."""
        config = DotProductAttentionConfig(
            hidden_dim=4,
            num_heads=2,
            init_weights="truncnormal002-identity",
            attn_impl=attn_impl if attn_impl != "sdpa" else None,
        )
        attn = DotProductAttention(config).to(device=gpu_device)
        assert torch.all(attn.proj.weight == 0)
        assert torch.all(attn.proj.bias == 0)

    # Skip CPU-only tests (already covered in test_dot_product.py)
    @pytest.mark.skip(reason="CPU-only tests are in test_dot_product.py")
    def test_invalid_dim_num_heads_gpu(self):
        pass

    @pytest.mark.skip(reason="CPU-only tests are in test_dot_product.py")
    def test_reset_parameters_invalid_gpu(self):
        pass


@pytest.mark.gpu
def test_attention_implementation_forward_path(device):
    """Test that GPU attention runs faster than CPU (basic check)."""
    torch.manual_seed(42)
    B, T, D = 4, 32, 16
    x = torch.randn(B, T, D, device=device)
    # causal mask for sdpa fallback
    mask = torch.zeros((T, T), device=device, dtype=torch.bool)
    mask = mask.masked_fill(torch.tril(torch.ones((T, T), device=device), diagonal=0) == 1, True)

    _attn_implentations = ["sdpa", "flash-attention-3", "kernels/flash-attn3"]

    results = []
    for is_causal in [False, True]:
        results.append((is_causal, {}))
        for _attn_impl in _attn_implentations:
            _dp_module = attention_module()
            _dp_module.attn_impl = _attn_impl  # Force SDPA for this test
            set_attn_impl(_attn_impl)
            import time
            start_time = time.time()
            y = _dp_module(x, is_causal=is_causal)
            fwd_time = time.time() - start_time
            results[-1][1][_attn_impl] = {
                "fwd_time": fwd_time,
                "output": y.detach().cpu(),  # Store output for potential further checks
            }

    _dp_module = attention_module()
    set_attn_impl("sdpa")
    start_time = time.time()
    y = _dp_module(x, attn_mask=mask, is_causal=False)
    sdpa_mask_time = time.time() - start_time

    for is_causal, data in results:
        sdpa_time = data["sdpa"]["fwd_time"]
        flash_time = data["flash-attention-3"]["fwd_time"]
        kernels_time = data["kernels/flash-attn3"]["fwd_time"]

        assert flash_time < sdpa_time, f"Flash Attention should be faster than SDPA (got {flash_time:.4f}s vs {sdpa_time:.4f}s)"
        assert kernels_time < sdpa_time, f"Kernels Flash Attention should be faster than SDPA (got {kernels_time:.4f}s vs {sdpa_time:.4f}s)"
        # NOTE: We don't assert flash_time < kernels_time because performance can vary based on implementation and hardware
        if is_causal:
            assert flash_time < sdpa_mask_time, f"Flash Attention should be faster than SDPA with mask (got {flash_time:.4f}s vs {sdpa_mask_time:.4f}s)"
            assert kernels_time < sdpa_mask_time, f"Kernels Flash Attention should be faster than SDPA with mask (got {kernels_time:.4f}s vs {sdpa_mask_time:.4f}s)"

        sdpa_out = data["sdpa"]["output"]
        flash_out = data["flash-attention-3"]["output"]
        kernels_out = data["kernels/flash-attn3"]["output"]

        assert torch.allclose(sdpa_out, flash_out, atol=1e-5), f"Flash Attention output should match SDPA (max diff: {(sdpa_out - flash_out).abs().max():.4e})"
        assert torch.allclose(sdpa_out, kernels_out, atol=1e-5), f"Kernels Flash Attention output should match SDPA (max diff: {(sdpa_out - kernels_out).abs().max():.4e})"

        if is_causal:
            assert torch.allclose(sdpa_out, y, atol=1e-5), f"SDPA with causal mask output should match SDPA without mask (max diff: {(sdpa_out - y).abs().max():.4e})"
            assert torch.allclose(y, flash_out, atol=1e-5), f"Flash Attention output should match SDPA with causal mask (max diff: {(sdpa_out - flash_out).abs().max():.4e})"
            assert torch.allclose(y, kernels_out, atol=1e-5), f"Kernels Flash Attention output should match SDPA with causal mask (max diff: {(sdpa_out - kernels_out).abs().max():.4e})"
    
