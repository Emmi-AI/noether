#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import pytest

from noether.core.schemas.dataset import AeroDataSpecs


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
                # hidden_dim is missing -> inject from parent
                # num_heads is missing -> inject from parent
                # mlp_expansion_factor is missing -> inject from parent
            },
        },
        "data_specs": AeroDataSpecs(
            position_dim=3,
            surface_output_dims={"loss_var": 1},
        ),
    }


@pytest.fixture
def base_ab_upt_config_dict():
    """Returns a minimal valid dictionary for AnchorBranchedUPTConfig."""
    return {
        "kind": "noether.modeling.models.ab_upt.AnchoredBranchedUPT",
        "name": "test_ab_upt_model",
        "hidden_dim": 128,
        "geometry_depth": 2,
        "physics_blocks": ["shared", "cross"],
        "num_surface_blocks": 2,
        "num_volume_blocks": 2,
        "supernode_pooling_config": {
            "input_dim": 3,
            "radius": 0.1,
            "max_degree": 32,
            # hidden_dim is missing, should be injected from parent
        },
        "transformer_block_config": {
            "num_heads": 4,
            "mlp_expansion_factor": 4,
            # hidden_dim is missing -> inject from parent
        },
        "data_specs": AeroDataSpecs(
            position_dim=3,
            surface_output_dims={"loss_var": 1},
        ),
    }


@pytest.fixture
def explicit_upt_config_dict():
    """Returns a UPTConfig dict with all fields explicitly set (pre-mixin behavior)."""
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
            "hidden_dim": 128,
        },
        "approximator_config": {
            "hidden_dim": 128,
            "num_heads": 4,
            "mlp_expansion_factor": 4,
        },
        "decoder_config": {
            "input_dim": 3,
            "depth": 1,
            "perceiver_block_config": {
                "num_heads": 4,
                "hidden_dim": 128,
                "mlp_expansion_factor": 2,
            },
        },
        "data_specs": AeroDataSpecs(
            position_dim=3,
            surface_output_dims={"loss_var": 1},
        ),
    }


@pytest.fixture
def explicit_ab_upt_config_dict():
    """Returns an AnchorBranchedUPTConfig dict with all fields explicitly set (pre-mixin behavior)."""
    return {
        "kind": "noether.modeling.models.ab_upt.AnchoredBranchedUPT",
        "name": "test_ab_upt_model",
        "hidden_dim": 128,
        "geometry_depth": 2,
        "physics_blocks": ["shared", "cross"],
        "num_surface_blocks": 2,
        "num_volume_blocks": 2,
        "supernode_pooling_config": {
            "input_dim": 3,
            "radius": 0.1,
            "max_degree": 32,
            "hidden_dim": 128,
        },
        "transformer_block_config": {
            "num_heads": 4,
            "mlp_expansion_factor": 4,
            "hidden_dim": 128,
        },
        "data_specs": AeroDataSpecs(
            position_dim=3,
            surface_output_dims={"loss_var": 1},
        ),
    }
