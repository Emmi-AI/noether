#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
from __future__ import annotations

import pytest
import torch
from torch import nn

from noether.core.optimizer.param_group_modifiers.weight_decay_by_name import WeightDecayByNameModifier
from noether.core.optimizer.schemas import ParamGroupModifierConfig


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(torch.randn(1))
        self.param2 = nn.Parameter(torch.randn(1))


def test_weight_decay_by_name_modifier_init():
    """Test the initialization of the modifier."""
    config = ParamGroupModifierConfig(name="test_param", value=0.01)
    modifier = WeightDecayByNameModifier(config)

    assert modifier.name == "test_param"
    assert modifier.value == 0.01
    assert not modifier.param_was_found


def test_get_properties_with_matching_name():
    """Test get_properties when the parameter name matches."""
    config = ParamGroupModifierConfig(name="param1", value=0.05)
    modifier = WeightDecayByNameModifier(config)
    model = SimpleModel()
    param = model.param1

    properties = modifier.get_properties(model, "param1", param)

    assert properties == {"weight_decay": 0.05}
    assert modifier.param_was_found


def test_get_properties_with_non_matching_name():
    """Test get_properties when the parameter name does not match."""
    config = ParamGroupModifierConfig(name="non_existent_param", value=0.05)
    modifier = WeightDecayByNameModifier(config)
    model = SimpleModel()
    param = model.param1

    properties = modifier.get_properties(model, "param1", param)

    assert properties == {}
    assert not modifier.param_was_found


def test_get_properties_raises_on_duplicate_name():
    """Test that finding the same parameter name twice raises an AssertionError."""
    config = ParamGroupModifierConfig(name="param1", value=0.05)
    modifier = WeightDecayByNameModifier(config)
    model = SimpleModel()
    param = model.param1

    # First call should succeed
    modifier.get_properties(model, "param1", param)
    assert modifier.param_was_found

    # Second call with the same name should fail
    with pytest.raises(RuntimeError, match="found two parameters matching name 'param1'"):
        modifier.get_properties(model, "param1", param)


def test_was_applied_successfully():
    """Test the was_applied_successfully method."""
    config = ParamGroupModifierConfig(name="param1", value=0.05)
    modifier = WeightDecayByNameModifier(config)
    model = SimpleModel()

    # Before finding the parameter
    assert not modifier.was_applied_successfully()

    # Find the parameter
    modifier.get_properties(model, "param1", model.param1)

    # After finding the parameter
    assert modifier.was_applied_successfully()


def test_str_representation():
    """Test the string representation of the modifier."""
    config = ParamGroupModifierConfig(name="cls_token", value=0.0)
    modifier = WeightDecayByNameModifier(config)
    expected_str = "WeightDecayByNameModifier(name=cls_token), value=0.0)"
    assert str(modifier) == expected_str
