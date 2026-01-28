#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch
import torch_geometric  # type: ignore[import-untyped]


def radius(x, y, r, max_num_neighbors, batch_x, batch_y):
    # Move tensors to CPU if on MPS device
    device = x.device
    if device.type == "mps":
        x = x.cpu()
        y = y.cpu()
        batch_x = batch_x.cpu() if batch_x is not None else None
        batch_y = batch_y.cpu() if batch_y is not None else None

    result = torch_geometric.nn.pool.radius(
        x,
        y,
        r,
        batch_x,
        batch_y,
        max_num_neighbors=max_num_neighbors,
    )

    # Move result back to MPS if original tensors were on MPS
    if device.type == "mps":
        result = result.to(device)

    return result


def knn(x, y, k, batch_x=None, batch_y=None):
    # Move tensors to CPU if on MPS device
    device = x.device
    if device.type == "mps":
        x = x.cpu()
        y = y.cpu()
        batch_x = batch_x.cpu() if batch_x is not None else None
        batch_y = batch_y.cpu() if batch_y is not None else None

    result = torch_geometric.nn.pool.knn(
        x=x,
        y=y,
        k=k,
        batch_x=batch_x,
        batch_y=batch_y,
    )

    # Move result back to MPS if original tensors were on MPS
    if device.type == "mps":
        result = result.to(device)

    return result


def segment_reduce(src, lengths, reduce):
    # segment_reduce is not implemented on MPS, so we move to CPU if needed
    device = src.device
    if device.type == "mps":
        src = src.cpu()
        lengths = lengths.cpu()

    result = torch.segment_reduce(
        src,
        reduce=reduce,
        lengths=lengths,
    )

    # Move result back to MPS if original tensors were on MPS
    if device.type == "mps":
        result = result.to(device)

    return result
