#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import pytest

from noether.core.schemas.modules.attention import AttentionPattern, JointAnchorAttentionConfig, TokenSpec
from noether.modeling.modules.attention.anchor_attention.joint import JointAnchorAttention


@pytest.fixture
def valid_config():
    return JointAnchorAttentionConfig(
        branches=["surface", "volume"],
        hidden_dim=16,
        num_heads=4,
        anchor_suffix="_anchors",  # Explicitly setting suffix to match TokenSpec expectations
    )


def test_init_raises_value_error_for_single_branch():
    """
    Test that initializing with fewer than 2 branches raises ValueError.
    Joint attention implies 'across' branches, so single branch is invalid.
    """
    config = JointAnchorAttentionConfig(
        branches=["surface"],
        hidden_dim=16,
        num_heads=4,
    )

    with pytest.raises(ValueError, match="requires at least two branches"):
        JointAnchorAttention(config)


def test_init_success(valid_config):
    """Test that initializing with a valid config (2 branches) works correctly."""
    model = JointAnchorAttention(valid_config)

    assert len(model.branches) == 2
    assert "surface" in model.branches
    assert "volume" in model.branches
    assert isinstance(model, JointAnchorAttention)


def test_create_attention_patterns_logic(valid_config):
    model = JointAnchorAttention(valid_config)

    token_specs = [
        TokenSpec(name="surface_anchors", size=100),
        TokenSpec(name="surface_queries", size=100),
        TokenSpec(name="volume_anchors", size=200),
        TokenSpec(name="volume_queries", size=200),
    ]
    patterns = model._create_attention_patterns(token_specs)
    assert len(patterns) == 1
    pattern = patterns[0]
    assert isinstance(pattern, AttentionPattern)

    expected_queries = {
        "surface_anchors",
        "surface_queries",
        "volume_anchors",
        "volume_queries",
    }
    assert set(pattern.query_tokens) == expected_queries

    expected_keys = {"surface_anchors", "volume_anchors"}
    assert set(pattern.key_value_tokens) == expected_keys


def test_pattern_ignores_irrelevant_tokens(valid_config):
    model = JointAnchorAttention(valid_config)

    token_specs = [
        TokenSpec(name="surface_queries", size=50),
        TokenSpec(name="volume_queries", size=50),
    ]

    patterns = model._create_attention_patterns(token_specs)
    pattern = patterns[0]

    assert set(pattern.query_tokens) == {"surface_queries", "volume_queries"}

    expected_keys = {"surface_anchors", "volume_anchors"}
    assert set(pattern.key_value_tokens) == expected_keys
