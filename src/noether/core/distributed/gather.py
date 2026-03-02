#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import einops
import torch
import torch.distributed as dist

from noether.core.distributed.config import get_world_size, is_distributed


def get_device_and_bfloat16supported():
    # gloo cpu -> okay
    # gloo cuda -> okay (although https://pytorch.org/docs/stable/distributed.html says it isn't supported)
    # nccl cpu -> fail (but gloo anyway recommended for cpu multiprocessing)
    # nccl cuda -> okay
    # bfloat16 cpu -> fail
    if not is_distributed():
        return torch.device("cpu"), True
    if dist.get_backend() == "nccl":
        return torch.device("cuda"), True
    if dist.get_backend() == "gloo":
        return torch.device("cpu"), False
    raise NotImplementedError


def get_bool_gather_supported():
    if not is_distributed():
        return True
    if dist.get_backend() == "nccl":
        return True
    if dist.get_backend() == "gloo":
        return False
    raise NotImplementedError


def _prepare_tensor(x):
    """
    prepare for distributed communication
    - wrap primitive types into tensors
    - push tensor onto supported device
    - convert bool to float if bool gathering is not supported
    - call .contiguous if x is not in a contiguous memory block
    """
    device, bfloat16_supported = get_device_and_bfloat16supported()
    # I think this doesn't work in some configuration not sure in which though
    # note in which configuration and convert back to bool after gather
    if isinstance(x, bool):
        raise RuntimeError
    if isinstance(x, float | int | list | tuple):
        x = torch.tensor(x, device=device)
        og_device = torch.device("cpu")
    else:
        og_device = x.device
    if x.dtype == torch.bfloat16 and not bfloat16_supported:
        x = x.type(torch.float32)
    # bool gather is not supported in some settings
    if x.dtype == torch.bool and not get_bool_gather_supported():
        x = x.type(torch.float32)
        to_bool = True
    else:
        to_bool = False
    if not x.is_contiguous():
        x = x.contiguous()
    return x.to(device), og_device, to_bool


def all_gather_grad(x, batch_dim=0):
    if not is_distributed():
        if x.ndim == 0:
            # distributed gather adds a dimension to scalars
            x = x.unsqueeze(0)
        return x

    x, _, to_bool = _prepare_tensor(x)
    if x.ndim == 0:
        result = torch.zeros(get_world_size(), device=x.device, dtype=x.dtype)
    else:
        result = torch.zeros(
            [i * get_world_size() if i == batch_dim else i for i in range(x.ndim)], device=x.device, dtype=x.dtype
        )
    dist.all_gather_into_tensor(result, x)

    if to_bool:
        result = result.bool()
    return result


@torch.no_grad()
def all_gather_nograd(x, batch_dim=0):
    all_gather_grad(x, batch_dim=batch_dim)


def all_gather_nograd_clipped(x, max_length):
    result = all_gather_nograd(x)
    if is_distributed():
        # gathering changes the order of the samples -> correct them
        # most of the time this is not noeeded (e.g. for metrics) as the order is not important
        # for things like predictions it does matter
        # 1 GPU: [0, 1, 2, 3, 4, 5, 6, 7]
        # 2 GPU: [0, 2, 4, 6] + [1, 3, 5, 7]
        # 4 GPU: [0, 4] + [1, 5] + [2, 6] + [3, 7]
        result = einops.rearrange(
            result,
            "(num_gpus len_per_gpu) ... -> (len_per_gpu num_gpus) ...",
            num_gpus=get_world_size(),
        )
        # DistributedSampler pads the dataset to give every GPU the same amount of samples
        return result[:max_length]
    return result


def all_reduce_sum_nograd(x):
    with torch.no_grad():
        return all_reduce_sum_grad(x)


def all_reduce_sum_grad(x):
    if not is_distributed():
        return x

    x, og_device, to_bool = _prepare_tensor(x)
    # all_reduce is differentiable https://github.com/pytorch/pytorch/issues/58005
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x = x.to(og_device)
    if to_bool:
        x = x.bool()
    return x


def reduce_mean_grad(x, dest_rank=0):
    x, og_device, to_bool = _prepare_tensor(x)
    if is_distributed():
        dist.reduce(x, dst=dest_rank, op=dist.ReduceOp.SUM)
        if dist.get_rank() == dest_rank:
            x = x / get_world_size()
    x = x.to(og_device)
    if to_bool:
        x = x.bool()
    return x


def reduce_mean_nograd(x, dest_rank=0):
    with torch.no_grad():
        return reduce_mean_grad(x, dest_rank=dest_rank)


def reduce_max_grad(x, dest_rank=0):
    if not is_distributed():
        return x
    x, og_device, to_bool = _prepare_tensor(x)
    dist.reduce(x, dst=dest_rank, op=dist.ReduceOp.MAX)
    x = x.to(og_device)
    if to_bool:
        x = x.bool()
    return x


def reduce_max_nograd(x, dest_rank=0):
    with torch.no_grad():
        return reduce_max_grad(x, dest_rank=dest_rank)


def all_reduce_mean_grad(x):
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
def all_reduce_mean_nograd(x):
    return all_reduce_mean_grad(x)
