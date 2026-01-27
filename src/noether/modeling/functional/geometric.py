#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import importlib.util

import torch

try:
    import torch_geometric  # type: ignore[import-untyped]

    HAS_PYG = True and importlib.util.find_spec("torch_cluster") is not None
except ImportError:
    HAS_PYG = False


def radius_pytorch(
    x: torch.Tensor,
    y: torch.Tensor,
    r: float,
    max_num_neighbors: int = 32,
    batch_x: torch.Tensor | None = None,
    batch_y: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fallback implementation of radius using pure PyTorch operations.

    Args:
        x: Source points (N, D).
        y: Query points (M, D).
        r: Radius to search for.
        max_num_neighbors: Maximum number of neighbors to return.
        batch_x: Batch index for source points.
        batch_y: Batch index for query points.

    Returns:
        Edge index (2, num_edges).
    """
    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.long)
    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.long)

    x_indices = torch.arange(x.size(0), device=x.device)
    y_indices = torch.arange(y.size(0), device=y.device)

    all_row = []
    all_col = []

    batches = torch.unique(batch_y)
    for b in batches:
        mask_x = batch_x == b
        mask_y = batch_y == b

        idx_x = x_indices[mask_x]
        idx_y = y_indices[mask_y]

        if idx_x.numel() == 0 or idx_y.numel() == 0:
            continue

        x_b = x[idx_x]
        y_b = y[idx_y]

        # dist: [N_y, N_x]
        dist = torch.cdist(y_b, x_b)

        within_radius = dist <= r

        y_idx, x_idx = torch.nonzero(within_radius, as_tuple=True)
        if y_idx.numel() == 0:
            continue

        _, y_idx_mapped, counts = torch.unique_consecutive(y_idx, return_inverse=True, return_counts=True)
        padded_cumsum = torch.zeros(counts.size(0) + 1, device=x.device, dtype=torch.long)
        padded_cumsum[1:] = torch.cumsum(counts, dim=0)

        local_idx = torch.arange(y_idx.size(0), device=x.device) - padded_cumsum[y_idx_mapped]
        max_neighbor_mask = local_idx < max_num_neighbors

        all_row.append(idx_y[y_idx[max_neighbor_mask]])
        all_col.append(idx_x[x_idx[max_neighbor_mask]])
    if not all_row:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)

    return torch.stack([torch.cat(all_row), torch.cat(all_col)], dim=0)


def knn_pytorch(
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
    batch_x: torch.Tensor | None = None,
    batch_y: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fallback implementation of knn using pure PyTorch operations.

    Args:
        x: Source points (N, D).
        y: Query points (M, D).
        k: Number of neighbors.
        batch_x: Batch index for source points.
        batch_y: Batch index for query points.

    Returns:
        Edge index (2, num_edges).
    """
    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.long)
    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.long)

    x_indices = torch.arange(x.size(0), device=x.device)
    y_indices = torch.arange(y.size(0), device=y.device)

    all_row = []
    all_col = []

    batches = torch.unique(batch_y)
    for b in batches:
        mask_x = batch_x == b
        mask_y = batch_y == b

        idx_x = x_indices[mask_x]
        idx_y = y_indices[mask_y]

        if idx_x.numel() == 0 or idx_y.numel() == 0:
            continue

        x_b = x[idx_x]
        y_b = y[idx_y]

        dist = torch.cdist(y_b, x_b)

        k_b = min(k, x_b.size(0))
        _, idx = dist.topk(k=k_b, dim=1, largest=False)

        row_b = torch.arange(y_b.size(0), device=x.device).view(-1, 1).expand(-1, k_b).flatten()
        col_b = idx.flatten()

        all_row.append(idx_y[row_b])
        all_col.append(idx_x[col_b])

    if not all_row:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)

    return torch.stack([torch.cat(all_row), torch.cat(all_col)], dim=0)


def radius(
    x: torch.Tensor,
    y: torch.Tensor,
    r: float,
    max_num_neighbors: int,
    batch_x: torch.Tensor | None = None,
    batch_y: torch.Tensor | None = None,
) -> torch.Tensor:
    if not HAS_PYG:
        return radius_pytorch(x, y, r, max_num_neighbors, batch_x, batch_y)

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


def knn(
    x: torch.Tensor, y: torch.Tensor, k: int, batch_x: torch.Tensor | None = None, batch_y: torch.Tensor | None = None
) -> torch.Tensor:
    if not HAS_PYG:
        return knn_pytorch(x, y, k, batch_x, batch_y)

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
