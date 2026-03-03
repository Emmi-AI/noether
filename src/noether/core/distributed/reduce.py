#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import torch
import torch.distributed as dist

from noether.core.distributed.config import get_world_size, is_distributed
from noether.core.distributed.gather import _prepare_tensor


def all_reduce_sum_nograd(x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return all_reduce_sum_grad(x)


def all_reduce_sum_grad(x: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return x

    x, og_device, to_bool = _prepare_tensor(x)
    # all_reduce is differentiable https://github.com/pytorch/pytorch/issues/58005
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x = x.to(og_device)
    if to_bool:
        x = x.bool()
    return x


def reduce_mean_grad(x: torch.Tensor, dest_rank=0) -> torch.Tensor:
    if not is_distributed():
        return x
    x, og_device, to_bool = _prepare_tensor(x)
    x /= get_world_size()
    dist.reduce(x, dst=dest_rank, op=dist.ReduceOp.SUM)
    x = x.to(og_device)
    if to_bool:
        x = x.bool()
    return x


def reduce_mean_nograd(x: torch.Tensor, dest_rank=0) -> torch.Tensor:
    with torch.no_grad():
        return reduce_mean_grad(x, dest_rank=dest_rank)


def reduce_max_grad(x: torch.Tensor, dest_rank=0) -> torch.Tensor:
    if not is_distributed():
        return x
    x, og_device, to_bool = _prepare_tensor(x)
    dist.reduce(x, dst=dest_rank, op=dist.ReduceOp.MAX)
    x = x.to(og_device)
    if to_bool:
        x = x.bool()
    return x


def reduce_max_nograd(x: torch.Tensor, dest_rank=0) -> torch.Tensor:
    with torch.no_grad():
        return reduce_max_grad(x, dest_rank=dest_rank)


def all_reduce_mean_grad(x: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return x
    x, og_device, to_bool = _prepare_tensor(x)
    # divide before all_reduce to avoid overflow in sum
    x /= get_world_size()
    x = all_reduce_sum_grad(x)
    x = x.to(og_device)
    if to_bool:
        x = x.bool()
    return x


@torch.no_grad()
def all_reduce_mean_nograd(x: torch.Tensor) -> torch.Tensor:
    return all_reduce_mean_grad(x)
