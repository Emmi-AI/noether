#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from noether.core.schemas.models.ab_upt import AnchorBranchedUPTConfig


def test_ab_upt_config_injects_hidden_dim(base_ab_upt_config_dict):
    """Test that hidden_dim is injected into both sub-configs from parent."""
    config = AnchorBranchedUPTConfig(**base_ab_upt_config_dict)

    # Check parent hidden_dim
    assert config.hidden_dim == 128

    # Check SupernodePoolingConfig injection
    assert config.supernode_pooling_config.hidden_dim == 128

    # Check TransformerBlockConfig injection
    assert config.transformer_block_config.hidden_dim == 128


def test_ab_upt_config_transformer_block_keeps_own_fields(base_ab_upt_config_dict):
    """Test that TransformerBlockConfig fields not in parent are preserved correctly."""
    config = AnchorBranchedUPTConfig(**base_ab_upt_config_dict)

    # TransformerBlockConfig should retain its own fields
    assert config.transformer_block_config.num_heads == 4
    assert config.transformer_block_config.mlp_expansion_factor == 4

    # These fields don't exist in parent, so they should not be affected by injection
    assert not hasattr(config, "num_heads")
    assert not hasattr(config, "mlp_expansion_factor")


def test_ab_upt_config_with_all_fields_explicit(explicit_ab_upt_config_dict):
    """Test backward compatibility when all fields are explicitly set (pre-mixin behavior)."""
    config = AnchorBranchedUPTConfig(**explicit_ab_upt_config_dict)

    # All values should match as explicitly provided
    assert config.hidden_dim == 128

    assert config.supernode_pooling_config.hidden_dim == 128

    assert config.transformer_block_config.hidden_dim == 128
    assert config.transformer_block_config.num_heads == 4
    assert config.transformer_block_config.mlp_expansion_factor == 4
