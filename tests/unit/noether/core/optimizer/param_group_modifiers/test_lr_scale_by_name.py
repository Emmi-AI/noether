#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
import pytest
from pydantic import ValidationError

from noether.core.optimizer.param_group_modifiers.lr_scale_by_name import LrScaleByNameModifier
from noether.core.optimizer.schemas import ParamGroupModifierConfig


def test_lr_scale_by_name_modifier_init_raises_error_if_scale_is_none():
    """Test that LrScaleByNameModifier raises a ValueError if scale is not provided."""
    with pytest.raises(ValidationError):
        ParamGroupModifierConfig(name="some_param")


def test_lr_scale_by_name_modifier_init_sets_attributes_correctly():
    """Test that LrScaleByNameModifier initializes its attributes correctly."""
    config = ParamGroupModifierConfig(name="some_param", scale=0.1)
    modifier = LrScaleByNameModifier(config)

    assert modifier.scale == 0.1
    assert modifier.name == "some_param"
    assert not modifier.param_was_found

    properties = modifier.get_properties(None, "some_other_para", None)
    assert properties == {}
    assert not modifier.param_was_found

    properties = modifier.get_properties(None, "some_param", None)
    assert modifier.param_was_found
    assert properties == {"lr_scale": 0.1}
    assert str(modifier) == "LrScaleByNameModifier(name=some_param,scale=0.1)"
    assert modifier.was_applied_successfully()

    with pytest.raises(ValueError, match="found two parameters matching name 'some_param'"):
        modifier.get_properties(None, "some_param", None)
