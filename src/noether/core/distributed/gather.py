#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import einops
import torch
import torch.distributed as dist
import torch.distributed.nn.functional

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


def _prepare_tensor(x: torch.Tensor):
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
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"Expected a tensor but got {type(x)}")
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


def all_gather_grad(x: torch.Tensor, batch_dim=0) -> torch.Tensor:
    if not is_distributed():
        if isinstance(x, torch.Tensor) and x.ndim == 0:
            # distributed gather adds a dimension to scalars
            x = x.unsqueeze(0)
        return x

    x, og_device, to_bool = _prepare_tensor(x)
    result = torch.distributed.nn.functional.all_gather(x)
    if result[0].ndim == 0:
        # scalars can't be concatenated
        result = [r.unsqueeze(0) for r in result]
    result = torch.concat(result, dim=batch_dim).to(og_device)

    if to_bool:
        result = result.bool()
    return result


@torch.no_grad()
def all_gather_nograd(x: torch.Tensor, batch_dim=0) -> torch.Tensor:
    return all_gather_grad(x, batch_dim=batch_dim)


def all_gather_nograd_clipped(x: torch.Tensor, max_length: int | None = None, batch_dim=0) -> torch.Tensor:
    result = all_gather_nograd(x, batch_dim=batch_dim)
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
        if max_length is not None:
            # DistributedSampler pads the dataset to give every GPU the same amount of samples
            return result[:max_length]
    return result
