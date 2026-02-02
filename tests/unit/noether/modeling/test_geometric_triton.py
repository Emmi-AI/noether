#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

"""Tests for Triton-accelerated geometric operations."""

import pytest
import torch

from noether.modeling.functional.geometric import HAS_TRITON, knn_pytorch, knn_triton, radius_pytorch, radius_triton


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestRadiusTriton:
    """Test Triton implementation of radius search."""

    def test_radius_triton_vs_pytorch(self):
        """Test that Triton implementation matches PyTorch fallback."""
        torch.manual_seed(42)

        # Create test data
        n_x, n_y, dim = 100, 50, 3
        x = torch.randn(n_x, dim, device="cuda", dtype=torch.float32)
        y = torch.randn(n_y, dim, device="cuda", dtype=torch.float32)
        r = 2.0
        max_neighbors = 10

        # Compute with Triton
        edges_triton = radius_triton(x, y, r, None, None, max_neighbors)

        # Compute with PyTorch fallback
        edges_pytorch = radius_pytorch(x, y, r, None, None, max_neighbors)

        # Both should have same number of edges
        assert edges_triton.size(1) == edges_pytorch.size(1)

        # Convert to sets for comparison (order may differ)
        edges_triton_set = {tuple(edge) for edge in edges_triton.t().cpu().tolist()}
        edges_pytorch_set = {tuple(edge) for edge in edges_pytorch.t().cpu().tolist()}

        assert edges_triton_set == edges_pytorch_set

    def test_radius_triton_batched(self):
        """Test Triton implementation with batched data."""
        torch.manual_seed(42)

        # Create batched test data
        n_x, n_y, dim = 60, 40, 3
        x = torch.randn(n_x, dim, device="cuda", dtype=torch.float32)
        y = torch.randn(n_y, dim, device="cuda", dtype=torch.float32)

        # Create batch indices (3 batches)
        batch_x = torch.cat(
            [
                torch.zeros(20, dtype=torch.long),
                torch.ones(20, dtype=torch.long),
                torch.full((20,), 2, dtype=torch.long),
            ]
        ).to("cuda")

        batch_y = torch.cat(
            [
                torch.zeros(15, dtype=torch.long),
                torch.ones(15, dtype=torch.long),
                torch.full((10,), 2, dtype=torch.long),
            ]
        ).to("cuda")

        r = 3.0
        max_neighbors = 5

        # Compute with Triton
        edges_triton = radius_triton(x, y, r, batch_x, batch_y, max_neighbors)

        # Compute with PyTorch fallback
        edges_pytorch = radius_pytorch(x, y, r, batch_x, batch_y, max_neighbors)

        # Verify edges respect batch boundaries
        for i in range(edges_triton.size(1)):
            y_idx = edges_triton[0, i].item()
            x_idx = edges_triton[1, i].item()
            assert batch_y[y_idx] == batch_x[x_idx]

        # Compare results
        assert edges_triton.size(1) == edges_pytorch.size(1)

    def test_radius_triton_empty_result(self):
        """Test Triton implementation with no neighbors found."""
        torch.manual_seed(42)

        # Create points far apart
        x = torch.tensor([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]], device="cuda")
        y = torch.tensor([[5.0, 5.0, 5.0]], device="cuda")
        r = 1.0
        max_neighbors = 10

        edges = radius_triton(x, y, r, None, None, max_neighbors)

        assert edges.size(1) == 0
        assert edges.shape == (2, 0)

    def test_radius_triton_max_neighbors(self):
        """Test that max_neighbors parameter is respected."""
        torch.manual_seed(42)

        # Create a cluster of points
        center = torch.zeros(3, device="cuda")
        x = center + torch.randn(50, 3, device="cuda") * 0.5
        y = center.unsqueeze(0)
        r = 2.0
        max_neighbors = 10

        edges = radius_triton(x, y, r, None, None, max_neighbors)

        # Should have at most max_neighbors edges
        assert edges.size(1) <= max_neighbors

    def test_knn_triton_vs_pytorch(self):
        """Test that Triton implementation matches PyTorch fallback."""
        torch.manual_seed(42)

        # Create test data
        n_x, n_y, dim = 10, 1, 3
        x = torch.randn(n_x, dim, device="cuda", dtype=torch.float32)
        y = torch.randn(n_y, dim, device="cuda", dtype=torch.float32)
        k = 15

        # Compute with Triton
        edges_triton = knn_triton(x, y, k=k)

        # Compute with PyTorch fallback
        edges_pytorch = knn_pytorch(x, y, k=k)

        # Both should have same number of edges
        assert edges_triton.size(1) == edges_pytorch.size(1)

        # Convert to sets for comparison (order may differ)
        edges_triton_set = {tuple(edge) for edge in edges_triton.t().cpu().tolist()}
        edges_pytorch_set = {tuple(edge) for edge in edges_pytorch.t().cpu().tolist()}

        assert edges_triton_set == edges_pytorch_set

    def test_knn_triton_batched(self):
        """Test KNN Triton implementation with batched data."""
        torch.manual_seed(42)

        # Create batched test data
        n_x, n_y, dim = 60, 40, 3
        x = torch.randn(n_x, dim, device="cuda", dtype=torch.float32)
        y = torch.randn(n_y, dim, device="cuda", dtype=torch.float32)

        # Create batch indices (3 batches)
        batch_x = torch.cat(
            [
                torch.zeros(20, dtype=torch.long),
                torch.ones(20, dtype=torch.long),
                torch.full((20,), 2, dtype=torch.long),
            ]
        ).to("cuda")

        batch_y = torch.cat(
            [
                torch.zeros(15, dtype=torch.long),
                torch.ones(15, dtype=torch.long),
                torch.full((10,), 2, dtype=torch.long),
            ]
        ).to("cuda")

        k = 5

        # Compute with Triton
        edges_triton = knn_triton(x, y, k=k, batch_x=batch_x, batch_y=batch_y)

        # Compute with PyTorch fallback
        edges_pytorch = knn_pytorch(x, y, k=k, batch_x=batch_x, batch_y=batch_y)

        # Compare results
        assert edges_triton.size(1) == edges_pytorch.size(1)

        # Check specific edges
        edges_triton_set = {tuple(edge) for edge in edges_triton.t().cpu().tolist()}
        edges_pytorch_set = {tuple(edge) for edge in edges_pytorch.t().cpu().tolist()}
        assert edges_triton_set == edges_pytorch_set

    def test_knn_triton_k_ge_nx(self):
        """Test KNN Triton when k >= number of points in a batch."""
        torch.manual_seed(42)

        n_x, n_y, dim = 5, 2, 3
        x = torch.randn(n_x, dim, device="cuda", dtype=torch.float32)
        y = torch.randn(n_y, dim, device="cuda", dtype=torch.float32)
        k = 10  # k > n_x

        # Compute with Triton
        edges_triton = knn_triton(x, y, k=k)

        # Compute with PyTorch fallback
        edges_pytorch = knn_pytorch(x, y, k=k)

        assert edges_triton.size(1) == edges_pytorch.size(1)

        edges_triton_set = {tuple(edge) for edge in edges_triton.t().cpu().tolist()}
        edges_pytorch_set = {tuple(edge) for edge in edges_pytorch.t().cpu().tolist()}
        assert edges_triton_set == edges_pytorch_set

        # Should return all n_x neighbors for each y
        assert edges_triton.size(1) == n_y * n_x

    def test_knn_triton_large_n(self):
        """Test KNN Triton with a larger number of points."""
        torch.manual_seed(42)

        n_x, n_y, dim = 1000, 100, 3
        x = torch.randn(n_x, dim, device="cuda", dtype=torch.float32)
        y = torch.randn(n_y, dim, device="cuda", dtype=torch.float32)
        k = 32

        # Compute with Triton
        edges_triton = knn_triton(x, y, k=k)

        # Compute with PyTorch fallback
        edges_pytorch = knn_pytorch(x, y, k=k)

        assert edges_triton.size(1) == edges_pytorch.size(1)
        # For large N, order might differ but sets should be same if distances are distinct
        # (randn distances are almost certainly distinct)
        edges_triton_set = {tuple(edge) for edge in edges_triton.t().cpu().tolist()}
        edges_pytorch_set = {tuple(edge) for edge in edges_pytorch.t().cpu().tolist()}
        assert edges_triton_set == edges_pytorch_set
