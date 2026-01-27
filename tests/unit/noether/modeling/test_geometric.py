#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import importlib.util

import pytest
import torch

try:
    import torch_geometric

    HAS_PYG_LIBS = True and importlib.util.find_spec("torch_cluster") is not None
except ImportError:
    HAS_PYG_LIBS = False

from noether.modeling.functional.geometric import knn_pytorch, radius_pytorch


@pytest.mark.skipif(not HAS_PYG_LIBS, reason="torch_geometric and torch_cluster are required for these tests")
class TestGeometricFallbacks:
    @pytest.fixture
    def sample_data(self):
        # Two batches of points
        # Batch 0: 5 points
        # Batch 1: 3 points
        x = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [10.0, 10.0, 10.0],
                [11.0, 10.0, 10.0],
                [20.0, 20.0, 20.0],
                [21.0, 20.0, 20.0],
                [22.0, 20.0, 20.0],
            ],
            dtype=torch.float,
        )
        batch_x = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.long)

        # Query points
        y = torch.tensor(
            [
                [0.5, 0.5, 0.0],
                [10.5, 10.5, 10.5],
                [21.0, 21.0, 21.0],
            ],
            dtype=torch.float,
        )
        batch_y = torch.tensor([0, 0, 1], dtype=torch.long)

        return x, y, batch_x, batch_y

    def sort_edges(self, edge_index, num_nodes):
        # Sort by row (y), then by col (x)
        perm = (edge_index[0] * (num_nodes + 1) + edge_index[1]).argsort()
        return edge_index[:, perm]

    def test_radius_pytorch_vs_pyg(self, sample_data):
        x, y, batch_x, batch_y = sample_data
        r = 2.0
        max_num_neighbors = 4

        # Fallback
        edge_index_fallback = radius_pytorch(x, y, r, max_num_neighbors, batch_x, batch_y)

        # PyG
        edge_index_pyg = torch_geometric.nn.pool.radius(x, y, r, batch_x, batch_y, max_num_neighbors=max_num_neighbors)

        edge_index_fallback = self.sort_edges(edge_index_fallback, x.size(0))
        edge_index_pyg = self.sort_edges(edge_index_pyg, x.size(0))

        assert torch.equal(edge_index_fallback, edge_index_pyg)

    def test_radius_pytorch_no_batch(self, sample_data):
        x, y, _, _ = sample_data
        r = 5.0
        max_num_neighbors = 10

        # Fallback
        edge_index_fallback = radius_pytorch(x, y, r, max_num_neighbors)

        # PyG
        edge_index_pyg = torch_geometric.nn.pool.radius(x, y, r, max_num_neighbors=max_num_neighbors)

        edge_index_fallback = self.sort_edges(edge_index_fallback, x.size(0))
        edge_index_pyg = self.sort_edges(edge_index_pyg, x.size(0))

        assert torch.equal(edge_index_fallback, edge_index_pyg)

    def test_knn_pytorch_vs_pyg(self, sample_data):
        x, y, batch_x, batch_y = sample_data
        k = 2

        # Fallback
        edge_index_fallback = knn_pytorch(x, y, k, batch_x, batch_y)

        # PyG
        edge_index_pyg = torch_geometric.nn.pool.knn(x, y, k, batch_x, batch_y)

        edge_index_fallback = self.sort_edges(edge_index_fallback, x.size(0))
        edge_index_pyg = self.sort_edges(edge_index_pyg, x.size(0))

        assert torch.equal(edge_index_fallback, edge_index_pyg)

    def test_knn_pytorch_no_batch(self, sample_data):
        x, y, _, _ = sample_data
        k = 5

        # Fallback
        edge_index_fallback = knn_pytorch(x, y, k)

        # PyG
        edge_index_pyg = torch_geometric.nn.pool.knn(x, y, k)

        edge_index_fallback = self.sort_edges(edge_index_fallback, x.size(0))
        edge_index_pyg = self.sort_edges(edge_index_pyg, x.size(0))

        assert torch.equal(edge_index_fallback, edge_index_pyg)
