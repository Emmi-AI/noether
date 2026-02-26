#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import importlib.util

import torch

HAS_PYG = importlib.util.find_spec("torch_cluster") is not None

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_X": 32}, num_warps=1),
            triton.Config({"BLOCK_X": 32}, num_warps=2),
            triton.Config({"BLOCK_X": 64}, num_warps=1),
            triton.Config({"BLOCK_X": 64}, num_warps=2),
            triton.Config({"BLOCK_X": 128}, num_warps=1),
            triton.Config({"BLOCK_X": 128}, num_warps=2),
            triton.Config({"BLOCK_X": 256}, num_warps=1),
            triton.Config({"BLOCK_X": 256}, num_warps=2),
            triton.Config({"BLOCK_X": 512}, num_warps=1),
            triton.Config({"BLOCK_X": 512}, num_warps=2),
            triton.Config({"BLOCK_X": 1024}, num_warps=1),
            triton.Config({"BLOCK_X": 1024}, num_warps=2),
        ],
        key=["n_x", "n_y", "dim"],  # Retune when these change
    )
    @triton.jit
    def _radius_kernel(
        x_ptr,
        y_ptr,
        batch_x_ptr,
        batch_y_ptr,
        edge_dst_ptr,
        n_x,
        n_y,
        dim: tl.constexpr,
        r_squared,
        max_neighbors: tl.constexpr,
        has_batch: tl.constexpr,
        BLOCK_X: tl.constexpr,
    ):
        """
        Optimized kernel: each program handles one y point, processes x in blocks.
        """
        y_idx = tl.program_id(0)

        if y_idx >= n_y:
            return

        # Load batch for y
        if has_batch:
            batch_y_val = tl.load(batch_y_ptr + y_idx)

        y_base = y_ptr + (y_idx * dim)

        write_idx = 0

        out_base = y_idx * max_neighbors

        # Process x points in blocks
        x_block_start = 0
        while x_block_start < n_x and write_idx < max_neighbors:
            x_offsets = x_block_start + tl.arange(0, BLOCK_X)
            x_mask = x_offsets < n_x

            batch_match = tl.full((BLOCK_X,), True, dtype=tl.int1)
            if has_batch:
                batch_x_vals = tl.load(batch_x_ptr + x_offsets, mask=x_mask, other=-1)
                batch_match = batch_x_vals == batch_y_val

            # Skip block entirely if no valid points
            if tl.sum(batch_match) > 0:
                # Compute distances for this block
                dist_sq = tl.zeros((BLOCK_X,), dtype=tl.float32)

                for d in tl.static_range(dim):
                    x_vals = tl.load(x_ptr + x_offsets * dim + d, mask=x_mask, other=0.0)
                    y_val = tl.load(y_base + d)
                    diff = x_vals - y_val
                    dist_sq += diff * diff

                within_radius = (dist_sq <= r_squared) & x_mask & batch_match

                valid_int = within_radius.to(tl.int32)
                cumsum = tl.cumsum(valid_int, axis=0)
                local_offsets = cumsum - valid_int  # Exclusive prefix sum
                block_count = tl.max(cumsum)  # Last element = total
                local_offsets = tl.cumsum(valid_int, axis=0) - valid_int

                # Global write position for each lane
                write_pos = out_base + write_idx + local_offsets

                # Only write if within max_num_neighbors limit
                can_write = within_radius & ((write_idx + local_offsets) < max_neighbors)
                tl.store(edge_dst_ptr + write_pos, x_offsets, mask=can_write)

                write_idx += block_count

            x_block_start += BLOCK_X

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_X": 32}, num_warps=1),
            triton.Config({"BLOCK_X": 32}, num_warps=2),
            triton.Config({"BLOCK_X": 64}, num_warps=1),
            triton.Config({"BLOCK_X": 64}, num_warps=2),
            triton.Config({"BLOCK_X": 128}, num_warps=1),
            triton.Config({"BLOCK_X": 128}, num_warps=2),
            triton.Config({"BLOCK_X": 256}, num_warps=1),
            triton.Config({"BLOCK_X": 256}, num_warps=2),
            triton.Config({"BLOCK_X": 512}, num_warps=1),
            triton.Config({"BLOCK_X": 512}, num_warps=2),
            triton.Config({"BLOCK_X": 1024}, num_warps=1),
        ],
        key=["n_x", "n_y", "dim"],  # Retune when these change
    )
    @triton.jit
    def _knn_kernel(
        x_ptr,
        y_ptr,
        batch_x_ptr,
        batch_y_ptr,
        edge_dst_ptr,
        dist_ptr,
        n_x,
        n_y,
        dim: tl.constexpr,
        k,
        has_batch: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_X: tl.constexpr,
    ):
        """
        Optimized kernel: each program handles one y point, processes x in blocks.
        Uses block-level pruning and BLOCK_K argmin passes instead of per-element extraction.
        """
        y_idx = tl.program_id(0)

        if y_idx >= n_y:
            return
        y_base = y_ptr + (y_idx * dim)

        # Load batch for y
        batch_y_val = tl.load(batch_y_ptr + y_idx) if has_batch else 0

        k_offs = tl.arange(0, BLOCK_K)

        top_dists = tl.full([BLOCK_K], float("inf"), dtype=tl.float32)
        # Mask out indices >= k so they are never selected as max (victim) during eviction
        top_dists = tl.where(k_offs < k, top_dists, -1.0)
        top_idxs = tl.full([BLOCK_K], -1, dtype=tl.int64)

        curr_max_dist = float("inf")
        x_offs = tl.arange(0, BLOCK_X)

        # Process x points in blocks
        for x_idx in tl.range(0, n_x, BLOCK_X):
            x_indices = x_idx + x_offs
            x_mask = x_indices < n_x

            # Compute distances for block (vectorized across x points)
            dist_sq = tl.zeros([BLOCK_X], dtype=tl.float32)
            for d in tl.static_range(dim):
                x_vals = tl.load(x_ptr + x_indices * dim + d, mask=x_mask, other=0.0)
                y_val = tl.load(y_base + d)
                diff = x_vals - y_val
                dist_sq += diff * diff

            # Apply validity mask: out-of-bounds and batch mismatches become inf
            if has_batch:
                batch_x_vals = tl.load(batch_x_ptr + x_indices, mask=x_mask, other=-1)
                valid = x_mask & (batch_x_vals == batch_y_val)
            else:
                valid = x_mask
            dist_sq = tl.where(valid, dist_sq, float("inf"))

            # Block-level pruning: skip entire block if no candidate can improve top-k
            if tl.min(dist_sq) < curr_max_dist:
                # Extract up to BLOCK_K best candidates using selection passes (argmin)
                for _ in tl.static_range(BLOCK_K):
                    cand_dist = tl.min(dist_sq)
                    cand_pos = tl.argmin(dist_sq, axis=0)
                    # Compute global x index directly from block start + position
                    cand_idx = (x_idx + cand_pos).to(tl.int64)

                    # Evict the worst top-k entry if candidate is better
                    worst_dist = tl.max(top_dists)
                    should_insert = cand_dist < worst_dist

                    is_victim = (top_dists == worst_dist) & (k_offs < k)
                    victim_pos = tl.argmax(is_victim.to(tl.int32), axis=0)

                    top_dists = tl.where((k_offs == victim_pos) & should_insert, cand_dist, top_dists)
                    top_idxs = tl.where((k_offs == victim_pos) & should_insert, cand_idx, top_idxs)
                    curr_max_dist = tl.max(top_dists)

                    # Remove this candidate so subsequent passes find the next best
                    dist_sq = tl.where(x_offs == cand_pos, float("inf"), dist_sq)

        # Write results (only k valid slots)
        out_mask = k_offs < k
        out_base = y_idx * k
        tl.store(dist_ptr + out_base + k_offs, top_dists, mask=out_mask)
        tl.store(edge_dst_ptr + out_base + k_offs, top_idxs, mask=out_mask)

except ImportError:
    HAS_TRITON = False


def radius_triton(
    x: torch.Tensor,
    y: torch.Tensor,
    r: float,
    batch_x: torch.Tensor | None = None,
    batch_y: torch.Tensor | None = None,
    max_num_neighbors: int | None = None,
) -> torch.Tensor:
    """
    Find all edges where points in x are within radius r of points in y.

    Args:
        x: Source points, shape (N, D) or flattened (N*D,) with D inferred
        y: Target points, shape (M, D) or flattened (M*D,)
        r: Radius threshold
        max_num_neighbors: Maximum neighbors per target point (default: N)
        batch_x: Batch indices for x points, shape (N,)
        batch_y: Batch indices for y points, shape (M,)

    Returns:
        Edge index tensor of shape (2, E) where:
        - row 0: source indices (from x)
        - row 1: target indices (from y)
    """
    assert x.device.type == "cuda", "Input must be on CUDA device"
    assert y.device.type == "cuda", "Input must be on CUDA device"

    # Handle input shapes
    if x.dim() == 1:
        # Assume same dim as y or infer from context
        if y.dim() == 2:
            dim = y.shape[1]
            n_x = x.shape[0] // dim
            x = x.view(n_x, dim)
        else:
            raise ValueError("Cannot infer dimensions from flat tensors")
    else:
        n_x, dim = x.shape

    if y.dim() == 1:
        n_y = y.shape[0] // dim
        y = y.view(n_y, dim)
    else:
        n_y = y.shape[0]
        assert y.shape[1] == dim, "x and y must have same dimension"

    # Ensure contiguous
    x = x.contiguous().float()
    y = y.contiguous().float()

    # Default max neighbors
    if max_num_neighbors is None:
        max_num_neighbors = n_x

    # Validate batch tensors
    if batch_x is not None and batch_y is not None:
        batch_x = batch_x.contiguous().int()
        batch_y = batch_y.contiguous().int()
        assert batch_x.shape[0] == n_x
        assert batch_y.shape[0] == n_y

    # Allocate output buffers
    max_total_edges = n_y * max_num_neighbors
    edge_dst = torch.full((max_total_edges,), fill_value=-1, dtype=torch.int64, device=x.device)

    r_squared = r * r

    # Launch kernel - one program per y point
    grid = (n_y,)

    _radius_kernel[grid](
        x,
        y,
        batch_x,
        batch_y,
        edge_dst,
        n_x,
        n_y,
        dim,
        r_squared,
        max_num_neighbors,
        batch_x is not None and batch_y is not None,
    )

    valid_mask = edge_dst != -1
    edge_src_valid = torch.arange(n_y, device=x.device).repeat_interleave(max_num_neighbors)[valid_mask]
    edge_dst_valid = edge_dst[valid_mask]

    return torch.stack([edge_src_valid, edge_dst_valid], dim=0)


def radius_pytorch(
    x: torch.Tensor,
    y: torch.Tensor,
    r: float,
    batch_x: torch.Tensor | None = None,
    batch_y: torch.Tensor | None = None,
    max_num_neighbors: int | None = None,
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
    if not x.dtype.is_floating_point:
        raise ValueError("radius search requires floating point tensors")

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
        # matrix multiplication approach is faster but suffers numerical issues
        # for vectors that are close together, because of catastrophic cancellation.
        dist = torch.cdist(y_b, x_b, compute_mode="donot_use_mm_for_euclid_dist")

        within_radius = dist <= r

        y_idx, x_idx = torch.nonzero(within_radius, as_tuple=True)
        if y_idx.numel() == 0:
            continue

        if max_num_neighbors is None:
            all_row.append(idx_y[y_idx])
            all_col.append(idx_x[x_idx])
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


def knn_triton(
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
    batch_x: torch.Tensor | None = None,
    batch_y: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Calculates k-nearest neighbors using a Triton kernel.

    Args:
        x: Reference points (N, D)
        y: Query points (M, D)
        k: Number of neighbors
        batch_x: Optional batch indices for x (N,)
        batch_y: Optional batch indices for y (M,)
        cosine: If True, uses Cosine distance. False = Euclidean.

    Returns:
        Indices of the k nearest neighbors for each y (M, K)
    """
    # 1. Input Validation and Shape checks
    assert x.dim() == 2 and y.dim() == 2, "Inputs must be 2D tensors"
    assert x.shape[1] == y.shape[1], "Feature dimension mismatch"

    # Ensure inputs are contiguous for simplicity
    x = x.contiguous()
    y = y.contiguous()

    n_x, d = x.shape
    n_y, _ = y.shape

    assert d <= 128, "This simple kernel supports D <= 128"

    # 2. Setup Batching
    has_batch = False
    if batch_x is not None and batch_y is not None:
        has_batch = True
        batch_x = batch_x.contiguous()
        batch_y = batch_y.contiguous()

    k = min(k, n_x)

    # 3. Allocate Output
    # We return indices (LongTensor) and Distances (FloatTensor)
    # The prompt requested returning "Tensor", usually indices are the goal of KNN.
    output_indices = torch.empty((n_y * k,), device=x.device, dtype=torch.int32)
    output_dists = torch.full((n_y, k), float("inf"), device=x.device, dtype=torch.float32)

    # 4. Launch Kernel
    # Grid size is simply the number of query points (one thread per y)
    grid = (n_y,)

    # Warning: Triton JIT requires constants for 'k' to generate static loops
    # If k changes dynamically, this will trigger recompilation.

    _knn_kernel[grid](
        x,
        y,
        batch_x,
        batch_y,
        output_indices,
        output_dists,
        n_x,
        n_y,
        d,
        k=k,
        has_batch=has_batch,
        BLOCK_K=triton.next_power_of_2(k),
        # IS_COSINE=cosine,
    )

    edge_src_valid = torch.arange(n_y, device=x.device).repeat_interleave(k)
    edge_dst_valid = output_indices

    if batch_x is not None and batch_y is not None:
        # small batches may have fewer than k neighbors, resulting in -1 entries
        valid_mask = edge_dst_valid != -1
        edge_src_valid = edge_src_valid[valid_mask]
        edge_dst_valid = edge_dst_valid[valid_mask]

    return torch.stack([edge_src_valid, edge_dst_valid], dim=0)


def knn_pytorch(
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
    batch_x: torch.Tensor | None = None,
    batch_y: torch.Tensor | None = None,
    cosine=False,
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

        if cosine:
            x_b = torch.nn.functional.normalize(x_b, p=2, dim=-1)
            y_b = torch.nn.functional.normalize(y_b, p=2, dim=-1)
            dist = 1 - torch.mm(y_b, x_b.t())
        else:
            dist = torch.cdist(y_b, x_b, compute_mode="donot_use_mm_for_euclid_dist")

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
    """Find all points within radius r.

    Args:
        x: Source points (N, D).
        y: Query points (M, D).
        r: Radius to search for.
        max_num_neighbors: Maximum number of neighbors to return per query.
        batch_x: Batch index for source points.
        batch_y: Batch index for query points.

    Returns:
        Edge index (2, num_edges).
        first row: source indices (from y)
        second row: target indices (from x)
    """
    # Try Triton if available and requested
    if not HAS_PYG and HAS_TRITON and x.device.type in ["cuda", "hip"]:
        return radius_triton(x, y, r, batch_x, batch_y, max_num_neighbors)

    # Try PyG if available
    if not HAS_PYG:
        return radius_pytorch(x, y, r, batch_x, batch_y, max_num_neighbors)

    import torch_geometric  # type: ignore[import-untyped]

    # Move tensors to CPU if on MPS device
    device = x.device
    if device.type == "mps":
        x = x.cpu()
        y = y.cpu()
        batch_x = batch_x.cpu() if batch_x is not None else None
        batch_y = batch_y.cpu() if batch_y is not None else None

    result: torch.Tensor = torch_geometric.nn.pool.radius(
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
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
    batch_x: torch.Tensor | None = None,
    batch_y: torch.Tensor | None = None,
) -> torch.Tensor:
    if not HAS_PYG and HAS_TRITON and x.device.type in ["cuda", "hip"]:
        return knn_triton(x, y, k, batch_x, batch_y)
    if not HAS_PYG:
        return knn_pytorch(x, y, k, batch_x, batch_y)

    import torch_geometric  # type: ignore[import-untyped]

    # Move tensors to CPU if on MPS device
    device = x.device
    if device.type == "mps":
        x = x.cpu()
        y = y.cpu()
        batch_x = batch_x.cpu() if batch_x is not None else None
        batch_y = batch_y.cpu() if batch_y is not None else None

    result: torch.Tensor = torch_geometric.nn.pool.knn(
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
