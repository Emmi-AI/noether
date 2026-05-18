#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

import pytest
import torch
from pydantic import ValidationError

from noether.core.schemas.modules.blocks import TransformerBlockConfig
from noether.modeling.modules.blocks.transformer import TransformerBlock

from .expected_output import DIT_BLOCK, TRANSFORMER_BLOCK

AttentionType = Literal[
    "dot_product",
    "perceiver",
    "transolver",
    "transolver_plusplus",
]


def test_transformer_block_forward():
    torch.manual_seed(42)
    # Define input parameters
    dim = 16
    num_heads = 4
    mlp_hidden_dim = 32
    drop_path = 0.0
    layerscale = 0.5
    eps = 1e-6
    init_weights = "truncnormal002"
    attention_constructor: AttentionType = "dot_product"

    # Create a TransformerBlock instance
    config = TransformerBlockConfig(  # type: ignore
        hidden_dim=dim,
        num_heads=num_heads,
        mlp_hidden_dim=mlp_hidden_dim,
        drop_path=drop_path,
        attention_constructor=attention_constructor,
        layerscale=layerscale,
        eps=eps,
        init_weights=init_weights,
    )
    transformer_block = TransformerBlock(config=config)

    # Create dummy input tensor
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, dim)

    # Create dummy attention mask
    attn_mask = torch.ones(seq_len, seq_len)
    attn_kwargs = {"attn_mask": attn_mask}

    # Perform forward pass
    output, kv_cache = transformer_block(x, attn_kwargs=attn_kwargs)

    # Assertions
    assert output.shape == x.shape, "Output shape mismatch"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert torch.allclose(output, TRANSFORMER_BLOCK, 1e-2), "Output value mismatch"
    assert kv_cache is None, "DotProductAttention should not return a KV cache"


def test_transformer_block_no_attn_mask():
    # Define input parameters
    dim = 32
    num_heads = 2
    mlp_hidden_dim = 64

    # Create a TransformerBlock instance
    config = TransformerBlockConfig(  # type: ignore
        hidden_dim=dim,
        num_heads=num_heads,
        mlp_hidden_dim=mlp_hidden_dim,
    )
    transformer_block = TransformerBlock(config=config)

    # Create dummy input tensor
    batch_size = 4
    seq_len = 8
    x = torch.randn(batch_size, seq_len, dim)

    # Perform forward pass without attention mask
    output, _ = transformer_block(x)

    # Assertions
    assert output.shape == x.shape, "Output shape mismatch"
    assert not torch.isnan(output).any(), "Output contains NaN values"


def test_transformer_block_default_mlp_hidden_dim():
    # Define input parameters
    dim = 16
    num_heads = 2

    # Create a TransformerBlock instance
    config = TransformerBlockConfig(  # type: ignore
        hidden_dim=dim,
        num_heads=num_heads,
        mlp_expansion_factor=4,
    )
    transformer_block = TransformerBlock(config=config)

    # Create dummy input tensor
    batch_size = 3
    seq_len = 5
    x = torch.randn(batch_size, seq_len, dim)

    # Perform forward pass
    output, _ = transformer_block(x)

    # Assertions
    assert output.shape == x.shape, "Output shape mismatch"
    assert not torch.isnan(output).any(), "Output contains NaN values"


def test_transformer_block_conditioned():
    dim = 4
    condition_dim = 32
    torch.manual_seed(0)
    config = TransformerBlockConfig(  # type: ignore
        hidden_dim=dim,
        num_heads=2,
        condition_dim=condition_dim,
        mlp_expansion_factor=4,
    )
    dit_block = TransformerBlock(config=config)
    batch_size = 3
    seq_len = 5
    x = torch.randn(batch_size, seq_len, dim)
    condition = torch.randn(batch_size, condition_dim)
    dit_output, _ = dit_block(x, condition=condition)
    assert dit_output.shape == x.shape, "Output shape mismatch"
    assert torch.allclose(DIT_BLOCK, dit_output, atol=1e-4)


def test_no_bias():
    config = TransformerBlockConfig(hidden_dim=8, num_heads=2, bias=False, mlp_expansion_factor=4)  # type: ignore
    block = TransformerBlock(config=config)
    for name, _ in block.named_parameters():
        assert not name.endswith(".bias")


def test_transformer_block_invalid_num_heads():
    """Test that invalid number of heads raises an error."""

    dim = 16
    num_heads = 5  # Not a divisor of dim

    with pytest.raises((ValueError, AssertionError)):
        config = TransformerBlockConfig(  # type: ignore
            hidden_dim=dim,
            num_heads=num_heads,
            mlp_expansion_factor=4,
        )
        transformer_block = TransformerBlock(config=config)


def test_transformer_block_zero_dim():
    """Test that zero hidden dimension raises an error."""

    with pytest.raises((ValueError, AssertionError, ValidationError)):
        config = TransformerBlockConfig(  # type: ignore
            hidden_dim=0,
            num_heads=2,
            mlp_expansion_factor=4,
        )
        transformer_block = TransformerBlock(config=config)


def test_transformer_block_gradient_flow():
    """Test that gradients flow through the transformer block."""
    dim = 16
    num_heads = 4

    config = TransformerBlockConfig(  # type: ignore
        hidden_dim=dim,
        num_heads=num_heads,
        mlp_expansion_factor=4,
    )
    transformer_block = TransformerBlock(config=config)

    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, dim, requires_grad=True)

    output, _ = transformer_block(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None, "Gradients should flow to input"
    assert not torch.isnan(x.grad).any(), "Gradients contain NaN values"
