#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import pytest

from noether.core.schemas.dataset import AeroDataSpecs
from noether.core.schemas.models.upt import UPTConfig


@pytest.fixture
def base_upt_config_dict():
    """Returns a minimal valid dictionary for UPTConfig."""
    return {
        "kind": "ab_upt",
        "name": "test_upt_model",
        "num_heads": 4,
        "hidden_dim": 128,
        "mlp_expansion_factor": 4,
        "approximator_depth": 2,
        "supernode_pooling_config": {
            "input_dim": 3,
            "radius": 0.1,
            "max_degree": 32,
            # hidden_dim is missing, should be injected from parent
        },
        "approximator_config": {
            # hidden_dim is missing -> inject from parent
            # num_heads is missing -> inject from parent
            # mlp_expansion_factor is missing -> inject from parent
        },
        "decoder_config": {
            "input_dim": 3,
            "depth": 1,
            "perceiver_block_config": {
                "num_heads": 4,
                "hidden_dim": 128,
                "mlp_expansion_factor": 4,
            },
        },
        "data_specs": AeroDataSpecs(
            position_dim=3,
            surface_output_dims={"loss_var": 1},
        ),
    }


def test_upt_config_injects_multiple_shared_fields(base_upt_config_dict):
    """Test that multiple shared fields (hidden_dim, num_heads, etc.) are injected."""
    config = UPTConfig(**base_upt_config_dict)

    # Check SupernodePoolingConfig injection (only has hidden_dim in common)
    assert config.supernode_pooling_config.hidden_dim == 128

    # Check ApproximatorConfig injection (shares multiple fields)
    # It should inherit all matching fields from UPTConfig
    assert config.approximator_config.hidden_dim == 128
    assert config.approximator_config.num_heads == 4
    assert config.approximator_config.mlp_expansion_factor == 4


def test_upt_config_respects_explicit_submodule_config(base_upt_config_dict):
    """Test that explicit values in sub-configs take precedence over parent config."""
    # Explicitly set shared fields in submodules to different values
    base_upt_config_dict["supernode_pooling_config"]["hidden_dim"] = 64

    base_upt_config_dict["approximator_config"] = {"hidden_dim": 256, "num_heads": 8, "mlp_expansion_factor": 2}

    config = UPTConfig(**base_upt_config_dict)

    # Parent remains unchanged
    assert config.hidden_dim == 128
    assert config.num_heads == 4

    # Submodules keep their explicit values
    assert config.supernode_pooling_config.hidden_dim == 64

    assert config.approximator_config.hidden_dim == 256
    assert config.approximator_config.num_heads == 8
    assert config.approximator_config.mlp_expansion_factor == 2
