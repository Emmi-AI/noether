#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch
from pydantic import ValidationError

from noether.core.schemas.modules.attention import AttentionConfig
from noether.modeling.modules.attention.dot_product import DotProductAttention, DotProductAttentionConfig

from .expected_outputs import DOT_PRODUCT_ATTENTION, DOT_PRODUCT_ATTENTION_WITH_MASK


@pytest.fixture
def attention_module():
    torch.manual_seed(42)
    config = DotProductAttentionConfig(hidden_dim=16, num_heads=4, init_weights="truncnormal002")
    return DotProductAttention(config)


def test_eval_mode():
    model = DotProductAttention(AttentionConfig(hidden_dim=16, num_heads=4, init_weights="truncnormal002"))
    model.eval()
    assert not model.training, "Model should be in eval mode"


def test_forward_shape(attention_module):
    torch.manual_seed(42)

    x = torch.randn(2, 10, 16)  # Batch size = 2, Sequence length = 10, Embedding dim = 16
    output = attention_module(x)
    assert output.shape == (2, 10, 16), "Output shape mismatch"
    assert attention_module.num_heads == 4, "Number of heads mismatch"
    assert attention_module.q.in_features == 16, "Input features mismatch"
    assert attention_module.q.out_features == 16, "Output features mismatch"
    assert attention_module.k.in_features == 16, "Input features mismatch"
    assert attention_module.k.out_features == 16, "Output features mismatch"
    assert attention_module.v.in_features == 16, "Input features mismatch"
    assert attention_module.v.out_features == 16, "Output features mismatch"
    assert attention_module.proj.in_features == 16, "Input features mismatch"
    assert attention_module.proj.out_features == 16, "Output features mismatch"
    assert torch.allclose(output, DOT_PRODUCT_ATTENTION, 1e-2), "Output is not as expected"

    output.sum().backward()
    assert attention_module.q.weight.grad is not None, "Gradients should not be None"
    assert attention_module.k.weight.grad is not None, "Gradients should not be None"
    assert attention_module.v.weight.grad is not None, "Gradients should not be None"
    assert attention_module.proj.weight.grad is not None, "Gradients should not be None"


def test_forward_with_mask(attention_module):
    torch.manual_seed(42)
    x = torch.randn(2, 10, 16)
    attn_mask = torch.zeros(10, 10)  # Sequence length = 10
    output = attention_module(x, attn_mask=attn_mask)
    assert output.shape == (2, 10, 16), "Output shape mismatch with attention mask"
    assert torch.allclose(output, DOT_PRODUCT_ATTENTION_WITH_MASK, 1e-2), "Output is not as expected"


def test_no_bias():
    config = DotProductAttentionConfig(hidden_dim=4, num_heads=2, bias=False)
    attn = DotProductAttention(config)
    assert attn.q.bias is None
    assert attn.k.bias is None
    assert attn.v.bias is None
    assert attn.proj.bias is None


def test_truncnormal_init0():
    config = DotProductAttentionConfig(hidden_dim=4, num_heads=2, init_weights="truncnormal002-identity")
    attn = DotProductAttention(config)
    assert torch.all(attn.proj.weight == 0)
    assert torch.all(attn.proj.bias == 0)


def test_invalid_dim_num_heads():
    with pytest.raises(ValueError, match="must be divisible by"):
        config = DotProductAttentionConfig(hidden_dim=15, num_heads=4)


def test_reset_parameters_invalid():
    with pytest.raises(ValidationError):
        config = DotProductAttentionConfig(hidden_dim=16, num_heads=4, init_weights="invalid")
        module = DotProductAttention(config)
