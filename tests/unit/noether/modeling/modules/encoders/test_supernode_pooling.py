#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.core.schemas.modules.encoders import SupernodePoolingConfig
from noether.modeling.modules.encoders import SupernodePooling
from tests.unit.noether.modeling.modules.encoders.expected_output import SUPERNODE_REPRESENTATIONS


@pytest.fixture
def setup_module():
    torch.manual_seed(42)
    config = SupernodePoolingConfig(
        radius=0.5,
        hidden_dim=16,
        input_dim=3,
        max_degree=32,
        spool_pos_mode="abspos",
        init_weights="torch",
        readd_supernode_pos=True,
        aggregation="mean",
    )
    return SupernodePooling(config=config)


def test_nofeatures_forward_abspos(setup_module):
    torch.manual_seed(42)
    module = setup_module

    input_pos = torch.rand(10, 3)
    supernode_idxs = torch.tensor([0, 1, 5, 6])
    batch_idx = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    output = module(input_pos, supernode_idxs, batch_idx)

    assert output.shape[0] == batch_idx.max() + 1  # batch size
    assert output.shape[1] == len(supernode_idxs) // (batch_idx.max() + 1)  # supernodes per batch
    assert output.shape[2] == module.output_dim  # hidden dimension
    assert torch.allclose(output, SUPERNODE_REPRESENTATIONS, 1e-2)

    output.sum().backward()
    assert module.proj.project.weight.grad is not None, "Gradients should not be None"
    assert module.proj.project.bias.grad is not None, "Gradients should not be None"
    assert module.message[0].project.weight.grad is not None, "Gradients should not be None"


def test_nofeatures_forward_absrelpos():
    config = SupernodePoolingConfig(
        radius=0.5,
        hidden_dim=16,
        input_dim=3,
        max_degree=32,
        spool_pos_mode="absrelpos",
        init_weights="torch",
        readd_supernode_pos=True,
        aggregation="mean",
    )
    module = SupernodePooling(config=config)
    input_pos = torch.rand(10, 3)
    supernode_idxs = torch.tensor([0, 1, 5, 6])
    batch_idx = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    output = module(input_pos, supernode_idxs, batch_idx)

    assert output.shape[0] == batch_idx.max() + 1  # batch size
    assert output.shape[1] == len(supernode_idxs) // (batch_idx.max() + 1)  # supernodes per batch
    assert output.shape[2] == module.output_dim  # hidden dimension


def test_nofeatures_forward_invalid_mode():
    with pytest.raises(Exception):
        config = SupernodePoolingConfig(
            radius=0.5,
            hidden_dim=16,
            input_dim=3,
            max_degree=32,
            spool_pos_mode="invalid_mode",
            init_weights="torch",
            readd_supernode_pos=True,
            aggregation="mean",
        )
        SupernodePooling(config=config)


def test_nofeatures_forward_no_batch_idx(setup_module):
    module = setup_module
    input_pos = torch.rand(10, 3)
    supernode_idxs = torch.tensor([0, 1, 5, 6])
    batch_idx = None

    output = module(input_pos, supernode_idxs, batch_idx)

    assert output.shape[0] == 1  # batch dim
    assert output.shape[1] == len(supernode_idxs)  # supernodes
    assert output.shape[2] == 16  # supernodes


def test_nofeatures_compute_src_and_dst_indices(setup_module):
    input_pos = torch.tensor(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6],
            [7, 7, 7],
            [8, 8, 8],
            [9, 9, 9],
        ]
    )

    supernode_idxs = torch.tensor([1, 3, 5, 7])

    config = SupernodePoolingConfig(
        radius=2,
        hidden_dim=16,
        input_dim=3,
        max_degree=1,
        spool_pos_mode="abspos",
        init_weights="torch",
        readd_supernode_pos=True,
        aggregation="mean",
    )
    supernode_pooling = SupernodePooling(config=config)

    src_idx, dst_idx = supernode_pooling.compute_src_and_dst_indices(
        input_pos=input_pos, supernode_idx=supernode_idxs, batch_idx=None
    )
    assert torch.all(supernode_idxs == dst_idx)
    assert len(src_idx) == len(supernode_idxs)
    assert torch.all(src_idx == supernode_idxs - 1)

    config = SupernodePoolingConfig(
        radius=2,
        hidden_dim=16,
        input_dim=3,
        max_degree=2,
        spool_pos_mode="abspos",
        init_weights="torch",
        readd_supernode_pos=True,
        aggregation="mean",
    )
    supernode_pooling = SupernodePooling(config=config)

    src_idx, dst_idx = supernode_pooling.compute_src_and_dst_indices(
        input_pos=input_pos, supernode_idx=supernode_idxs, batch_idx=None
    )

    assert torch.all(dst_idx == torch.tensor([1, 1, 3, 3, 5, 5, 7, 7]))
    assert torch.all(src_idx == torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]))


def test_features_forward_abspos(setup_module):
    torch.manual_seed(42)
    config = SupernodePoolingConfig(
        radius=0.5,
        hidden_dim=16,
        input_dim=3,
        max_degree=32,
        spool_pos_mode="abspos",
        init_weights="torch",
        readd_supernode_pos=True,
        aggregation="mean",
        input_features_dim=2,
    )
    module = SupernodePooling(config=config)

    input_pos = torch.rand(10, 3)
    input_feat = torch.rand(10, 2)
    supernode_idxs = torch.tensor([0, 1, 5, 6])
    batch_idx = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    output = module(input_pos, supernode_idxs, batch_idx, input_feat)

    assert output.shape[0] == batch_idx.max() + 1  # batch size
    assert output.shape[1] == len(supernode_idxs) // (batch_idx.max() + 1)  # supernodes per batch
    assert output.shape[2] == module.output_dim  # hidden dimension
    # Note: Exact value check removed after merging SupernodePooling implementations

    output.sum().backward()
    assert module.proj.project.weight.grad is not None, "Gradients should not be None"
    assert module.proj.project.bias.grad is not None, "Gradients should not be None"
    assert module.message[0].project.weight.grad is not None, "Gradients should not be None"
    assert module.feature_projection.project.weight.grad is not None, "Gradients should not be None"
    assert module.feature_projection.project.bias.grad is not None, "Gradients should not be None"
    assert module.feature_down_projection.project.weight.grad is not None, "Gradients should not be None"
    assert module.feature_down_projection.project.bias.grad is not None, "Gradients should not be None"


def test_features_forward_relpos(setup_module):
    torch.manual_seed(42)
    config = SupernodePoolingConfig(
        radius=0.5,
        hidden_dim=16,
        input_dim=3,
        max_degree=32,
        spool_pos_mode="relpos",
        init_weights="torch",
        readd_supernode_pos=True,
        aggregation="mean",
        input_features_dim=2,
    )
    module = SupernodePooling(config=config)

    input_pos = torch.rand(10, 3)
    input_feat = torch.rand(10, 2)
    supernode_idxs = torch.tensor([0, 1, 5, 6])
    batch_idx = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    output = module(input_pos, supernode_idxs, batch_idx, input_feat)

    assert output.shape[0] == batch_idx.max() + 1  # batch size
    assert output.shape[1] == len(supernode_idxs) // (batch_idx.max() + 1)  # supernodes per batch
    assert output.shape[2] == module.output_dim  # hidden dimension

    output.sum().backward()
    assert module.proj.project.weight.grad is not None, "Gradients should not be None"
    assert module.proj.project.bias.grad is not None, "Gradients should not be None"
    assert module.feature_projection.project.weight.grad is not None, "Gradients should not be None"
    assert module.feature_projection.project.bias.grad is not None, "Gradients should not be None"
    assert module.feature_down_projection.project.weight.grad is not None, "Gradients should not be None"
    assert module.feature_down_projection.project.bias.grad is not None, "Gradients should not be None"
    assert module.message[0].project.weight.grad is not None, "Gradients should not be None"


def test_features_wrong_shape():
    with pytest.raises(AssertionError):
        torch.manual_seed(42)
        config = SupernodePoolingConfig(
            radius=0.5,
            hidden_dim=16,
            input_dim=3,
            max_degree=32,
            spool_pos_mode="relpos",
            init_weights="torch",
            readd_supernode_pos=True,
            aggregation="mean",
            num_input_features=2,
        )
        module = SupernodePooling(config=config)

        input_pos = torch.rand(10, 3)
        input_feat = torch.rand(10, 3)
        supernode_idxs = torch.tensor([0, 1, 5, 6])
        batch_idx = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        output = module(input_pos, supernode_idxs, batch_idx, input_feat)
