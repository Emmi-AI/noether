#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import time

import pytest
import torch

try:
    import torch_cluster  # noqa: F401
    import torch_geometric  # noqa: F401

    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from noether.modeling.functional.geometric import knn, knn_pytorch, radius_pytorch, radius_triton


def measure_runtime(func, *args, num_runs=10, **kwargs):
    # Warmup
    for _ in range(3):
        func(*args, **kwargs)

    if torch.cuda.is_available() and args[0].is_cuda:
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_runs):
            func(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event) / num_runs / 1000.0  # to seconds
    else:
        start_time = time.perf_counter()
        for _ in range(num_runs):
            func(*args, **kwargs)
        end_time = time.perf_counter()
        return (end_time - start_time) / num_runs


# @pytest.mark.skip
class TestGeometricPerformance:
    @pytest.fixture
    def large_sample_data(self, request):
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

    @pytest.mark.parametrize("large_sample_data", ["cpu", "cuda", "mps"], indirect=True)
    def test_performance_radius(self, large_sample_data):
        x, y, batch_x, batch_y = large_sample_data
        r = 0.5
        max_num_neighbors = 32

        device = x.device.type

        # Measure fallback
        time_fallback = measure_runtime(radius_pytorch, x, y, r, max_num_neighbors, batch_x, batch_y)

        # Measure PyG (via the wrapper which uses HAS_PYG=True)
        time_pyg = measure_runtime(torch_geometric.nn.pool.radius, x, y, r, batch_x, batch_y, max_num_neighbors)

        time_triton = (
            measure_runtime(radius_triton, x, y, r, max_num_neighbors, batch_x, batch_y)
            if device == "cuda"
            else float("inf")
        )

        print(
            f"\nRadius Performance ({device}): Fallback={time_fallback: .6f}s, PyG={time_pyg: .6f}s , Triton={time_triton: .6f}s"
        )
        # No assertion on speed as fallback is expected to be slower, but we can log results

    @pytest.mark.parametrize("large_sample_data", ["cpu", "cuda", "mps"], indirect=True)
    def test_performance_knn(self, large_sample_data):
        x, y, batch_x, batch_y = large_sample_data
        k = 16

        device = x.device.type

        # Measure fallback
        time_fallback = measure_runtime(knn_pytorch, x, y, k, batch_x, batch_y)

        # Measure PyG
        time_pyg = measure_runtime(knn, x, y, k, batch_x, batch_y)

        print(f"\nKNN Performance ({device}): Fallback={time_fallback: .6f}s, PyG={time_pyg: .6f}s")
