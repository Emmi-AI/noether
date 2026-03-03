#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import os
from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from noether.core.distributed import (
    all_gather_grad,
    all_gather_nograd,
    all_gather_nograd_clipped,
    all_reduce_mean_grad,
    all_reduce_sum_grad,
    get_rank,
    get_world_size,
)


def _dist_worker(rank, world_size, fn, args):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    if torch.cuda.is_available():
        backend = "nccl"
        torch.cuda.set_device(rank)
    else:
        backend = "gloo"

    dist.init_process_group(backend=backend, init_method="env://", world_size=world_size, rank=rank)
    try:
        fn(*args)
    finally:
        dist.destroy_process_group()


@pytest.fixture
def run_distributed():
    # skip if less than 2 GPUs are available for NCCL backend
    if torch.cuda.is_available() and torch.cuda.device_count() < 2:
        pytest.skip("Distributed tests require at least 2 GPUs for NCCL backend")

    def _run(fn, world_size=2, args=()):
        # add cwd to path so that the worker can import the test module
        import sys

        sys.path.append(os.getcwd())
        mp.spawn(_dist_worker, args=(world_size, fn, args), nprocs=world_size)

    return _run


def _check_all_gather_grad_rank_2():
    rank = get_rank()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # rank 0: [1, 1], rank 1: [2, 2]
    source = torch.full((2,), fill_value=float(rank + 1), device=device, requires_grad=True)

    gathered = all_gather_grad(source)
    assert gathered.shape == (4,)

    expected = torch.tensor([1.0, 1.0, 2.0, 2.0], device=device)
    assert torch.allclose(gathered, expected)

    gathered.sum().backward()
    # For all gather the gradient gets multiplied by the world size
    # because each rank calls backward and accumulates the gradients
    assert torch.allclose(source.grad, get_world_size() * torch.tensor([1.0, 1.0], device=device))


def test_all_gather_grad_rank_2(run_distributed):
    run_distributed(_check_all_gather_grad_rank_2, world_size=2)


def _check_all_reduce_sum_grad_rank_2():
    rank = get_rank()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # rank 0: [1, 1], rank 1: [2, 2]
    source = torch.full((2,), fill_value=float(rank + 1), device=device, requires_grad=True)

    # Expected result: [3, 3] (1+2, 1+2)
    reduced_sum = all_reduce_sum_grad(source)
    assert torch.all(reduced_sum == 3.0)

    reduced_sum.sum().backward()
    # d(source_0 + source_1)/d(source_rank) = 1
    # When backward is called on all ranks, gradients accumulate: 1.0 * world_size = 2.0
    assert torch.allclose(source.grad, torch.tensor([2.0, 2.0], device=device))


def test_all_reduce_sum_grad_rank_2(run_distributed):
    run_distributed(_check_all_reduce_sum_grad_rank_2, world_size=2)


def _check_all_reduce_mean_grad_rank_2():
    rank = get_rank()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # rank 0: [1, 1], rank 1: [2, 2]
    source = torch.full((2,), fill_value=float(rank + 1), device=device, requires_grad=True)

    # Expected result: [1.5, 1.5] ((1+2)/2)
    reduced_mean = all_reduce_mean_grad(source)
    assert torch.all(reduced_mean == 1.5)

    reduced_mean.sum().backward()
    # d((source_0 + source_1)/2)/d(source_rank) = 0.5
    # When backward is called on all ranks, gradients accumulate: 0.5 * world_size = 1.0
    assert torch.allclose(source.grad, torch.tensor([1.0, 1.0], device=device))


def test_all_reduce_mean_grad_rank_2(run_distributed):
    run_distributed(_check_all_reduce_mean_grad_rank_2, world_size=2)


def _check_all_gather_nograd_clipped_rank_2():
    rank = get_rank()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # rank 0: [1, 1], rank 1: [2, 2] -> [1, 1, 2, 2]
    source = torch.full((2,), fill_value=float(rank + 1), device=device, requires_grad=True)

    # clipped to max_length=3 -> [1, 1, 2]
    clipped = all_gather_nograd_clipped(source, max_length=3)
    assert clipped.shape == (3,)

    expected = torch.tensor([1.0, 2.0, 1.0], device=device)
    assert torch.allclose(clipped, expected)
    assert clipped.grad_fn is None


def test_all_gather_nograd_clipped_rank_2(run_distributed):
    run_distributed(_check_all_gather_nograd_clipped_rank_2, world_size=2)


def _check_distributed_ops_generic(op, has_grad, is_gather):
    world_size = get_world_size()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    source = torch.randn(5, 6, device=device)
    source_clone = source.clone().requires_grad_(True)

    # * 5 to make a computation graph
    source_with_graph = source_clone * 5
    x = op(source_with_graph)

    assert torch.is_tensor(x)
    assert x.dtype == torch.float32

    if is_gather:
        # Check specific condition for clipped gather
        is_clipped = isinstance(op, partial) and "clipped" in op.func.__name__

        if is_clipped:
            # partial(all_gather_nograd_clipped, max_length=1)
            assert x.shape == (1, 6)
        else:
            # Standard gather: concatenates along dim 0
            assert x.shape == (world_size * 5, 6)
    else:
        # For reduce, shape remains the same
        assert x.shape == (5, 6)

    # Gradient checks
    if has_grad:
        assert x.requires_grad
        assert x.grad_fn is not None

        x.sum().backward()
        assert source_clone.grad is not None
    else:
        assert not x.requires_grad
        assert x.grad_fn is None


@pytest.mark.parametrize(
    ("op", "has_grad", "is_gather"),
    [
        (all_gather_grad, True, True),
        (all_gather_nograd, False, True),
        (partial(all_gather_nograd_clipped, max_length=1), False, True),
        (all_reduce_sum_grad, True, False),
        (all_reduce_mean_grad, True, False),
    ],
)
def test_distributed_ops_generic(op, has_grad, is_gather, run_distributed):
    run_distributed(_check_distributed_ops_generic, world_size=2, args=(op, has_grad, is_gather))
