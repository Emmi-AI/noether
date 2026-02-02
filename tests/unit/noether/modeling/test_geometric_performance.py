#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

try:
    import torch_geometric

    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from noether.modeling.functional.geometric import knn_pytorch, radius_pytorch, radius_triton


def wrap_with_sync(func, device):
    """Wrap function to include CUDA synchronization if needed."""

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if device == "cuda":
            torch.cuda.synchronize()
        return result

    return wrapper


@pytest.mark.skipif(not HAS_PYG, reason="torch_geometric and torch_cluster are required for these tests")
class TestGeometricPerformance:
    """Performance tests for geometric operations."""

    @pytest.fixture
    def large_sample_data(self, request):
        """Fixture providing large datasets for benchmarking.

        Args:
            request: Pytest request object containing the device to use.

        Returns:
            Tuple of (x, y, batch_x, batch_y) tensors on the specified device.
        """
        device = request.param
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if device == "mps" and not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        torch.manual_seed(42)
        num_points = 100_000
        num_queries = 8_000

        x = torch.randn(num_points, 3, device=device)
        y = torch.randn(num_queries, 3, device=device)

        # 5 batches
        batch_x = torch.randint(0, 5, (num_points,), device=device).sort()[0]
        batch_y = torch.randint(0, 5, (num_queries,), device=device).sort()[0]

        return x, y, batch_x, batch_y

    @pytest.mark.benchmark(group="radius")
    @pytest.mark.parametrize("large_sample_data", ["cpu", "cuda", "mps"], indirect=True)
    @pytest.mark.parametrize("implementation", ["fallback", "pyg", "triton"])
    def test_performance_radius(self, benchmark, large_sample_data, implementation):
        """Benchmark radius search implementations.

        Args:
            benchmark: Pytest benchmark fixture.
            large_sample_data: Fixture providing dataset.
            implementation: Implementation to benchmark ('fallback', 'pyg', or 'triton').
        """
        x, y, batch_x, batch_y = large_sample_data
        r = 0.5
        max_num_neighbors = 32
        device = x.device.type

        if implementation == "triton" and device != "cuda":
            pytest.skip("Triton implementation only supported on CUDA")

        if implementation == "pyg" and device == "mps":
            pytest.skip("torch_geometric radius not supported on MPS")

        if implementation == "fallback":
            func = radius_pytorch
            args = (x, y, r, max_num_neighbors, batch_x, batch_y)
        elif implementation == "pyg":
            func = torch_geometric.nn.pool.radius
            args = (x, y, r, batch_x, batch_y, max_num_neighbors)
        elif implementation == "triton":
            func = radius_triton
            args = (x, y, r, max_num_neighbors, batch_x, batch_y)

        benchmark(wrap_with_sync(func, device), *args)

    @pytest.mark.benchmark(group="knn")
    @pytest.mark.parametrize("large_sample_data", ["cpu", "cuda", "mps"], indirect=True)
    @pytest.mark.parametrize("implementation", ["fallback", "pyg"])
    def test_performance_knn(self, benchmark, large_sample_data, implementation):
        """Benchmark KNN search implementations.

        Args:
            benchmark: Pytest benchmark fixture.
            large_sample_data: Fixture providing dataset.
            implementation: Implementation to benchmark ('fallback' or 'pyg').
        """
        x, y, batch_x, batch_y = large_sample_data
        k = 16
        device = x.device.type

        if implementation == "pyg" and device == "mps":
            pytest.skip("torch_geometric radius not supported on MPS")

        if implementation == "fallback":
            func = knn_pytorch
            args = (x, y, k, batch_x, batch_y)
        elif implementation == "pyg":
            func = torch_geometric.nn.pool.knn
            args = (x, y, k, batch_x, batch_y)

        benchmark(wrap_with_sync(func, device), *args)
