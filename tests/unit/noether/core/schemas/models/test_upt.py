#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from noether.core.schemas.models.upt import UPTConfig


def test_upt_config_injects_multiple_shared_fields(base_upt_config_dict):
    """Test that multiple shared fields (hidden_dim, num_heads, etc.) are injected."""
    config = UPTConfig(**base_upt_config_dict)

    # Parent fields should match as explicitly provided
    assert config.hidden_dim == 128
    assert config.num_heads == 4
    assert config.mlp_expansion_factor == 4

    # Check SupernodePoolingConfig injection
    assert config.supernode_pooling_config.hidden_dim == 128

    # Check ApproximatorConfig injection
    assert config.approximator_config.hidden_dim == 128
    assert config.approximator_config.num_heads == 4
    assert config.approximator_config.mlp_expansion_factor == 4

    # Check DecoderConfig's PerceiverBlockConfig injection (two-level injection)
    assert config.decoder_config.perceiver_block_config.hidden_dim == 128
    assert config.decoder_config.perceiver_block_config.num_heads == 4
    assert config.decoder_config.perceiver_block_config.mlp_expansion_factor == 4


def test_upt_config_with_all_fields_explicit(explicit_upt_config_dict):
    """Test backward compatibility when all fields are explicitly set (pre-mixin behavior)."""
    config = UPTConfig(**explicit_upt_config_dict)

    # Parent fields should match as explicitly provided
    assert config.hidden_dim == 128
    assert config.num_heads == 4
    assert config.mlp_expansion_factor == 4

    # SupernodePoolingConfig should retain its own explicitly set hidden_dim
    assert config.supernode_pooling_config.hidden_dim == 128

    # ApproximatorConfig should retain its own explicitly set fields
    assert config.approximator_config.hidden_dim == 128
    assert config.approximator_config.num_heads == 4
    assert config.approximator_config.mlp_expansion_factor == 4

    # DecoderConfig's PerceiverBlockConfig should retain its own explicitly set fields
    assert config.decoder_config.perceiver_block_config.hidden_dim == 128
    assert config.decoder_config.perceiver_block_config.num_heads == 4
    assert config.decoder_config.perceiver_block_config.mlp_expansion_factor == 2
