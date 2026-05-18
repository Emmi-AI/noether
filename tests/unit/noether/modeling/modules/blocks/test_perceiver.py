#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch
from torch import nn

from noether.core.schemas.modules.blocks import PerceiverBlockConfig
from noether.modeling.modules.attention import PerceiverAttention
from noether.modeling.modules.blocks.perceiver import PerceiverBlock
from noether.modeling.modules.layers import UnquantizedDropPath
from noether.modeling.modules.mlp import UpActDownMlp

from .expected_output import DIT_PERCEIVER_BLOCK, PERCEIVER_BLOCK


def test_perceiver_block_initialization():
    dim = 64
    num_heads = 8
    kv_dim = 32
    mlp_hidden_dim = 128
    drop_path = 0.1
    norm_ctor = nn.RMSNorm
    eps = 1e-5
    init_weights = "truncnormal002"

    config = PerceiverBlockConfig(
        hidden_dim=dim,
        num_heads=num_heads,
        kv_dim=kv_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        drop_path=drop_path,
        normalization_constructor=norm_ctor,
        eps=eps,
        init_weights=init_weights,
    )
    block = PerceiverBlock(config=config)

    # Check if the attributes are correctly initialized
    assert isinstance(block.norm1q, nn.RMSNorm)
    assert block.norm1q.normalized_shape == (dim,)
    assert block.norm1q.eps == eps

    assert isinstance(block.norm1kv, nn.RMSNorm)
    assert block.norm1kv.normalized_shape == (kv_dim,)
    assert block.norm1kv.eps == eps

    assert isinstance(block.attn, PerceiverAttention)

    assert isinstance(block.drop_path1, UnquantizedDropPath)
    assert block.drop_path1.drop_prob == drop_path

    assert isinstance(block.norm2, nn.RMSNorm)
    assert block.norm2.normalized_shape == (dim,)
    assert block.norm2.eps == eps

    assert isinstance(block.mlp, UpActDownMlp)
    # Check that mlp has correct layer structure
    assert block.mlp.fc1.in_features == dim
    assert block.mlp.fc1.out_features == mlp_hidden_dim
    assert block.mlp.fc2.in_features == mlp_hidden_dim
    assert block.mlp.fc2.out_features == dim

    assert isinstance(block.drop_path2, UnquantizedDropPath)
    assert block.drop_path2.drop_prob == drop_path


def test_perceiver_block_default_initialization():
    dim = 64
    num_heads = 8

    config = PerceiverBlockConfig(hidden_dim=dim, num_heads=num_heads, mlp_expansion_factor=4)
    block = PerceiverBlock(config=config)

    # Check default values
    assert block.norm1kv.normalized_shape == (dim,)
    assert block.mlp.fc1.out_features == dim * 4  # hidden_dim
    assert block.drop_path1.drop_prob == 0.0
    assert block.drop_path2.drop_prob == 0.0


def test_perceiver_block_forward():
    torch.manual_seed(42)
    dim = 16
    num_heads = 4
    kv_dim = 8
    mlp_hidden_dim = 128
    drop_path = 0.1
    norm_ctor = nn.RMSNorm
    eps = 1e-5
    init_weights = "truncnormal002"

    config = PerceiverBlockConfig(
        hidden_dim=dim,
        num_heads=num_heads,
        kv_dim=kv_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        drop_path=drop_path,
        normalization_constructor=norm_ctor,
        eps=eps,
        init_weights=init_weights,
    )
    block = PerceiverBlock(config=config)

    batch_size = 2
    seq_len = 10

    q = torch.randn(batch_size, seq_len, dim)
    kv = torch.randn(batch_size, seq_len, kv_dim)
    attn_mask = torch.ones(seq_len, seq_len)

    output, _ = block(q, kv, attn_kwargs=dict(attn_mask=attn_mask))

    assert output.shape == q.shape
    assert not torch.isnan(output).any()
    assert torch.allclose(output, PERCEIVER_BLOCK, 1e-2)


# @pytest.mark.skip(reason="Assertion error: config.kv_dim is None")
def test_perceiver_block_conditioned():
    dim = 4
    condition_dim = 32
    torch.manual_seed(0)
    config = PerceiverBlockConfig(
        hidden_dim=dim,
        num_heads=2,
        condition_dim=condition_dim,
        mlp_expansion_factor=4,
    )
    block = PerceiverBlock(config=config)
    batch_size = 3
    seq_len = 5
    x = torch.randn(batch_size, seq_len, dim)
    condition = torch.randn(batch_size, condition_dim)
    dit_output, _ = block(q=x, kv=x, condition=condition)
    assert dit_output.shape == x.shape, "Output shape mismatch"
    assert torch.allclose(DIT_PERCEIVER_BLOCK, dit_output, atol=1e-4)


def test_perceiver_block_conditioned_with_kv_dim():
    """Test that conditioning works when kv_dim != hidden_dim."""
    hidden_dim = 8
    kv_dim = 4
    condition_dim = 32
    torch.manual_seed(0)
    config = PerceiverBlockConfig(
        hidden_dim=hidden_dim,
        num_heads=2,
        kv_dim=kv_dim,
        condition_dim=condition_dim,
        mlp_expansion_factor=4,
    )
    block = PerceiverBlock(config=config)
    batch_size = 2
    seq_len = 5
    q = torch.randn(batch_size, seq_len, hidden_dim)
    kv = torch.randn(batch_size, seq_len, kv_dim)
    condition = torch.randn(batch_size, condition_dim)
    output, _ = block(q=q, kv=kv, condition=condition)
    assert output.shape == q.shape, "Output shape mismatch"
    assert not torch.isnan(output).any(), "Output contains NaN"


def test_no_bias():
    config = PerceiverBlockConfig(hidden_dim=8, num_heads=2, bias=False, mlp_expansion_factor=4)
    block = PerceiverBlock(config=config)
    assert not any(name.endswith(".bias") for name, _ in block.named_parameters()), "There should be no bias parameters"
