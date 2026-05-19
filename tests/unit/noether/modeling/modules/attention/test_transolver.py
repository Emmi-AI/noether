#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.modeling.modules.attention.transolver import TransolverAttention, TransolverAttentionConfig
from noether.modeling.modules.layers import LinearProjection

from .expected_outputs import TRANSOLVER_ATTENTION


@pytest.mark.parametrize(
    ("dim", "num_heads", "num_slices", "dropout", "bias", "init_weights"),
    [
        (64, 8, 4, 0.1, False, "truncnormal002"),
        (128, 16, 8, 0.0, True, "torch"),
        (256, 32, 16, 0.2, False, "truncnormal"),
    ],
)
def test_transolver_attention_init(dim, num_heads, num_slices, dropout, bias, init_weights):
    config = TransolverAttentionConfig(
        hidden_dim=dim,
        num_heads=num_heads,
        num_slices=num_slices,
        dropout=dropout,
        bias=bias,
        init_weights=init_weights,
    )
    model = TransolverAttention(config)

    # Check basic attributes
    assert model.num_heads == num_heads
    assert model.dropout == dropout
    assert model.temperature.shape == (1, num_heads, 1, 1)
    assert torch.allclose(model.temperature, torch.full((1, num_heads, 1, 1), 0.5))

    # Check LinearProjection layers
    assert isinstance(model.in_project_x, LinearProjection)
    assert isinstance(model.in_project_fx, LinearProjection)
    assert isinstance(model.in_project_slice, LinearProjection)
    assert isinstance(model.q, LinearProjection)
    assert isinstance(model.k, LinearProjection)
    assert isinstance(model.v, LinearProjection)
    assert isinstance(model.proj, LinearProjection)

    # Check Dropout layer
    assert isinstance(model.proj_dropout, torch.nn.Dropout)
    assert model.proj_dropout.p == dropout


def test_forward_transolver_attention():
    torch.manual_seed(42)
    config = TransolverAttentionConfig(hidden_dim=16, num_heads=8, num_slices=4)
    attention_module = TransolverAttention(config)
    input = torch.randn(2, 10, 16)
    out = attention_module(x=input, attn_mask=None)
    assert out.shape == (2, 10, 16)
    assert torch.allclose(out, TRANSOLVER_ATTENTION, 1e-2)

    out.sum().backward()

    assert attention_module.in_project_fx.project.weight.grad is not None
    assert attention_module.in_project_fx.project.weight.grad is not None
    assert attention_module.in_project_slice.project.weight.grad is not None
    assert attention_module.q.project.weight.grad is not None
    assert attention_module.k.project.weight.grad is not None
    assert attention_module.v.project.weight.grad is not None
    assert attention_module.proj.project.weight.grad is not None

    assert attention_module.in_project_fx.project.bias.grad is not None
    assert attention_module.in_project_fx.project.bias.grad is not None
    assert attention_module.in_project_slice.project.bias.grad is not None
    assert attention_module.proj.project.bias.grad is not None


def test_forward_transolver_attention_create_slices():
    batch_size = 2
    num_input_points = 10
    num_slices = 4
    heads = 8
    dim = 64
    config = TransolverAttentionConfig(hidden_dim=dim, num_heads=heads, num_slices=num_slices)
    attention_module = TransolverAttention(config)
    input = torch.randn(batch_size, num_input_points, dim)
    slice_token, slice_weights = attention_module.create_slices(x=input, num_input_points=10, attn_mask=None)
    assert slice_token.shape == (
        batch_size,
        heads,
        num_slices,
        dim // heads,
    )  # (batch_size, num_heads, num_slices, dim//num_heads)

    assert slice_weights.shape == (batch_size, heads, num_input_points, num_slices)


def test_transolver_attention_invalid_init_weights():
    with pytest.raises(ValueError):
        config = TransolverAttentionConfig(hidden_dim=64, num_heads=8, num_slices=4, init_weights="invalid")
        TransolverAttention(config)
