#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import os
import sys
from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from noether.core.distributed import get_rank, get_world_size
from noether.data.stats import RunningMoments


_WORLD_SIZE = 2

_cuda_skip = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD_SIZE,
    reason="Requires at least 2 CUDA GPUs",
)


def _dist_worker_loop(rank, world_size, device, task_queue, result_queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"  # different port from test_distributed.py
    if device == "cuda":
        backend = "nccl"
        torch.cuda.set_device(rank)
    else:
        backend = "gloo"
    dist.init_process_group(backend=backend, init_method="env://", world_size=world_size, rank=rank)
    try:
        while True:
            item = task_queue.get()
            if item is None:
                break
            fn, args = item
            try:
                fn(*args)
                result_queue.put(None)
            except Exception as e:
                result_queue.put(e)
    finally:
        dist.destroy_process_group()


@pytest.fixture(
    scope="module",
    params=["cpu", pytest.param("cuda", marks=_cuda_skip)],
)
def run_distributed(request):
    device = request.param
    sys.path.append(os.getcwd())
    ctx = mp.get_context("spawn")
    task_queues = [ctx.Queue() for _ in range(_WORLD_SIZE)]
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(target=_dist_worker_loop, args=(rank, _WORLD_SIZE, device, task_queues[rank], result_queue))
        for rank in range(_WORLD_SIZE)
    ]
    for p in processes:
        p.start()

    def _run(fn, args=()):
        for q in task_queues:
            q.put((fn, (device,) + args))
        errors = [result_queue.get() for _ in range(_WORLD_SIZE)]
        errors = [e for e in errors if e is not None]
        if errors:
            raise errors[0]

    yield _run

    for q in task_queues:
        q.put(None)
    for p in processes:
        p.join()


# Shared reference data: 100 rows × 3 features, deterministic across both workers.
def _full_data() -> torch.Tensor:
    return torch.arange(300, dtype=torch.float64).reshape(100, 3)


def _check_all_reduce_matches_single_pass(device):
    rank = get_rank()
    world = get_world_size()
    full = _full_data().to(device)

    # Sequential reference (computed identically on both ranks).
    expected = RunningMoments().to(device)
    expected.push_tensor(full)

    # Per-rank shard: every world-th row.
    shard = full[rank::world]
    actual = RunningMoments().to(device)
    actual.push_tensor(shard)
    actual.all_reduce_()

    assert actual.count == expected.count
    assert torch.allclose(actual.mean, expected.mean, atol=1e-12)
    assert torch.allclose(actual.std, expected.std, atol=1e-12)
    assert torch.allclose(actual._min, expected._min, atol=1e-12)
    assert torch.allclose(actual._max, expected._max, atol=1e-12)


def test_all_reduce_matches_single_pass(run_distributed):
    run_distributed(_check_all_reduce_matches_single_pass)


def _check_all_reduce_handles_empty_rank(device):
    """If one rank has zero samples, the global moments still match a single-pass over the other rank's data."""
    rank = get_rank()
    full = _full_data().to(device)
    shard = full if rank == 0 else full[:0]

    expected = RunningMoments().to(device)
    expected.push_tensor(full)

    actual = RunningMoments().to(device)
    if shard.size(0) > 0:
        actual.push_tensor(shard)
    actual.all_reduce_()

    assert actual.count == expected.count
    assert torch.allclose(actual.mean, expected.mean, atol=1e-12)
    assert torch.allclose(actual.std, expected.std, atol=1e-12)


def test_all_reduce_handles_empty_rank(run_distributed):
    run_distributed(_check_all_reduce_handles_empty_rank)
