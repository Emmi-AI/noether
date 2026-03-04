#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from functools import partial

import pytest
import torch

from noether.core.distributed import (
    all_gather_grad,
    all_gather_nograd,
    all_gather_nograd_clipped,
    all_reduce_mean_grad,
    all_reduce_sum_grad,
)


@pytest.mark.parametrize(
    ("op", "has_grad"),
    [
        (all_gather_grad, True),
        (all_gather_nograd, False),
        (partial(all_gather_nograd_clipped, max_length=None), False),
        (all_reduce_sum_grad, True),
        (all_reduce_mean_grad, True),
    ],
)
def test_tensor(op, has_grad):
    source = torch.randn(5, 6)
    source_clone = source.clone().requires_grad_(True)
    # * 5 to make a computation graph
    source_with_graph = source_clone * 5
    x = op(source_with_graph)

    assert torch.is_tensor(x)
    assert x.shape == (5, 6)
    assert x.dtype == torch.float32
    assert torch.all(source_with_graph == x)
    assert not has_grad or x.requires_grad == source_with_graph.requires_grad

    if has_grad:
        x.sum().backward()
        assert torch.all(torch.full_like(source, fill_value=5.0) == source_clone.grad)
