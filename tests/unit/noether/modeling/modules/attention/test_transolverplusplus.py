#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.modeling.modules.attention import TransolverPlusPlusAttention
from noether.modeling.modules.attention.transolver_plusplus import TransolverPlusPlusAttentionConfig


@pytest.mark.parametrize("dim", [256])
@pytest.mark.parametrize("heads", [8])
@pytest.mark.parametrize("dim_head", [64])
@pytest.mark.parametrize("dropout", [0.0])
@pytest.mark.parametrize("slice_num", [64])
@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("num_points", [1000])
def test_physics_attention_plus_plus_forward_pass_shape(
    dim: int,
    heads: int,
    dim_head: int,
    dropout: float,
    slice_num: int,
    batch_size: int,
    num_points: int,
) -> None:
    config = TransolverPlusPlusAttentionConfig(
        hidden_dim=dim,
        num_heads=heads,
        num_slices=slice_num,
        dropout=dropout,
    )
    model = TransolverPlusPlusAttention(config)
    data = torch.randn(batch_size, slice_num, dim)
    out = model(data)

    assert out is not None, "Output should not be None"
    assert out.shape == (batch_size, slice_num, dim), "Output shape should be (batch_size, slice_num, dim)"
