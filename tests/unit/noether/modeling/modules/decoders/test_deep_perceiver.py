#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.modeling.modules.blocks.perceiver import PerceiverBlockConfig
from noether.modeling.modules.decoders import DeepPerceiverDecoder
from noether.modeling.modules.decoders.deep_perceiver import DeepPerceiverDecoderConfig


@pytest.fixture
def decoder():
    torch.manual_seed(42)
    perceiver_block_config = PerceiverBlockConfig(
        hidden_dim=16,
        num_heads=4,
        mlp_expansion_factor=4,
    )
    config = DeepPerceiverDecoderConfig(
        perceiver_block_config=perceiver_block_config,
        input_dim=3,
        depth=2,
    )
    return DeepPerceiverDecoder(config=config)


def test_initialization(decoder):
    assert isinstance(decoder.blocks, torch.nn.ModuleList)
    assert len(decoder.blocks) == 2


def test_forward_valid_input(decoder):
    torch.manual_seed(42)
    batch_size = 2
    num_latent_tokens = 4
    num_output_pos = 2

    kv = torch.randn(batch_size, num_latent_tokens, 16)
    queries = torch.randn(batch_size, num_output_pos, 16)

    output = decoder(kv=kv, queries=queries)
    print(output)
    # assert torch.allclose(output, DEEP_PERCEIVER_DECODER, 2e-4)
    # assert output.shape == (batch_size, num_output_pos, 16)
    assert isinstance(output, torch.Tensor)


def test_forward_invalid_input_shape(decoder):
    kv = torch.randn(2, 16)  # Invalid shape
    queries = torch.randn(2, 8, 16)

    with pytest.raises(AssertionError):
        decoder(kv=kv, queries=queries)


def test_forward_invalid_queries_shape(decoder):
    kv = torch.randn(2, 16, 16)
    queries = torch.randn(2, 8)  # Invalid shape

    with pytest.raises(AssertionError):
        decoder(kv=kv, queries=queries)
