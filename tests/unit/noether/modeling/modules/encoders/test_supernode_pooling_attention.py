#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.core.schemas.modules.encoders import SupernodePoolingConfig
from noether.modeling.modules.encoders import SupernodePooling


def test_attention_pooling():
    torch.manual_seed(42)
    config = SupernodePoolingConfig(
        radius=10.0,  # Large radius to ensure connectivity
        hidden_dim=16,
        input_dim=3,
        aggregation="attention",
        num_heads=4,
        spool_pos_mode="abspos",
        init_weights="torch",
    )
    module = SupernodePooling(config=config)

    num_points = 10
    input_pos = torch.rand(num_points, 3)
    supernode_idxs = torch.tensor([0, 1, 5, 6])
    batch_idx = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    output = module(input_pos, supernode_idxs, batch_idx)

    # Batch size 2. Supernodes [0, 1] and [5, 6]. Total 2 per batch.
    assert output.shape == (2, 2, 16)

    # Test gradients flow
    loss = output.sum()
    loss.backward()

    assert module.attn_weights.project.weight.grad is not None
    assert module.attn_weights.project.bias.grad is not None


def test_attention_pooling_invalid_heads():
    with pytest.raises(ValueError, match="must be divisible"):
        config = SupernodePoolingConfig(
            radius=0.5,
            hidden_dim=16,
            input_dim=3,
            aggregation="attention",
            num_heads=3,  # 16 not divisible by 3
        )
        SupernodePooling(config=config)


def test_attention_pooling_default_heads():
    # num_heads defaults to 1 if not specified but aggregation=attention
    # config defines num_heads=1 by default valid pydantic
    config = SupernodePoolingConfig(
        radius=0.5,
        hidden_dim=16,
        input_dim=3,
        aggregation="attention",
        # num_heads default is 1
    )
    module = SupernodePooling(config=config)
    assert module.num_heads == 1
    assert module.attn_weights.project.out_features == 1
