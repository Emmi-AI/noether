#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.core.schemas.modules.attention import PerceiverAttentionConfig
from noether.modeling.modules.attention.perceiver import PerceiverAttention

from .expected_outputs import PERCEIVER_ATTENTION


def test_perceiver_attention_init_valid():
    # Test valid initialization
    dim = 64
    kv_dim = 32
    num_heads = 8
    init_weights = "truncnormal002"

    config = PerceiverAttentionConfig(hidden_dim=dim, kv_dim=kv_dim, num_heads=num_heads, init_weights=init_weights)
    attention = PerceiverAttention(config)

    assert attention.num_heads == num_heads
    assert attention.head_dim == dim // num_heads
    assert attention.init_weights == init_weights
    assert attention.k.in_features == kv_dim
    assert attention.k.out_features == dim
    assert attention.v.in_features == kv_dim
    assert attention.v.out_features == dim
    assert attention.q.in_features == dim
    assert attention.q.out_features == dim
    assert attention.proj.in_features == dim
    assert attention.proj.out_features == dim


def test_perceiver_attention_init_invalid_dim_num_heads():
    # Test invalid dim not divisible by num_heads
    with pytest.raises(ValueError, match="The 'hidden_dim' must be divisible by 'num_heads'"):
        config = PerceiverAttentionConfig(hidden_dim=65, num_heads=8)
        PerceiverAttention(config)


def test_perceiver_attention_init_invalid_init_weights():
    # Test invalid initialization method
    with pytest.raises(ValueError):
        config = PerceiverAttentionConfig(hidden_dim=64, num_heads=8, init_weights="invalid_method")
        PerceiverAttention(config)


def test_no_bias():
    config = PerceiverAttentionConfig(hidden_dim=4, num_heads=2, bias=False)
    attn = PerceiverAttention(config)
    assert attn.q.bias is None
    assert attn.k.bias is None
    assert attn.v.bias is None
    assert attn.proj.bias is None


def test_truncnormal_init0():
    config = PerceiverAttentionConfig(hidden_dim=4, num_heads=2, init_weights="truncnormal002-identity")
    attn = PerceiverAttention(config)
    assert torch.all(attn.proj.weight == 0)
    assert torch.all(attn.proj.bias == 0)


def test_perceiver_attention_forward_shape():
    # Test forward pass shape
    torch.manual_seed(42)

    dim = 16
    kv_dim = 8
    num_heads = 4
    init_weights = "truncnormal002"

    config = PerceiverAttentionConfig(hidden_dim=dim, kv_dim=kv_dim, num_heads=num_heads, init_weights=init_weights)
    attention = PerceiverAttention(config)

    q = torch.randn(2, 10, dim)
    kv = torch.randn(2, 10, kv_dim)
    output, new_cache = attention(q, kv)

    assert torch.allclose(output, PERCEIVER_ATTENTION, 1e-2)
    assert new_cache is not None

    output.sum().backward()
    assert output.shape == (2, 10, dim)

    assert attention.k.weight.grad is not None
    assert attention.v.weight.grad is not None
    assert attention.q.weight.grad is not None
    assert attention.proj.weight.grad is not None
    assert attention.k.bias.grad is not None
    assert attention.v.bias.grad is not None
    assert attention.q.bias.grad is not None
    assert attention.proj.bias.grad is not None

    kv = torch.randn(2, 10 * 2, kv_dim)
    output, _ = attention(q, kv)
    assert output.shape == (2, 10, dim)

    q = torch.randn(2, 10 * 2, dim)
    output, _ = attention(q, kv)
    assert output.shape == (2, 10 * 2, dim)
