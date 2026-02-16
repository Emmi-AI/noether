#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from unittest.mock import MagicMock, call, patch

import pytest
import torch

from noether.core.initializers.previous_run import PreviousRunInitializer
from noether.core.models.composite import CompositeModel
from noether.core.models.model import Model

MODULE_PATH = "noether.core.initializers.previous_run"


@pytest.fixture
def mock_config():
    """Creates a mock configuration object."""
    config = MagicMock()
    config.run_id = "run_123"
    config.model_name = "test_model"
    config.load_optim = False
    config.model_info = None
    config.pop_ckpt_kwargs_keys = list()
    config.stage_name = "train"
    config.checkpoint = "latest"

    # PreviousRunInitializer specific fields
    config.keys_to_remove = list()
    config.patterns_to_remove = list()
    config.patterns_to_rename = list()
    config.patterns_to_instantiate = list()
    return config


@pytest.fixture
def initializer(mock_config):
    """Creates an instance of PreviousRunInitializer with mocked dependencies."""
    with patch("noether.core.initializers.previous_run.CheckpointInitializer.__init__"):
        init = PreviousRunInitializer(initializer_config=mock_config)
        # Manually set attributes usually set by super().__init__
        init.logger = MagicMock()
        return init


@pytest.fixture
def mock_model():
    """Creates a mock Model."""
    model = MagicMock(spec=Model)
    model.name = "my_model"
    # Setup state_dict for patterns_to_instantiate tests:
    model.state_dict.return_value = {"layer.weight": torch.tensor([1.0])}
    return model


def test_init_weights_raises_type_error(initializer):
    """Test that passing a non-model raises TypeError."""
    with pytest.raises(TypeError, match="can only initialize Model or CompositeModel"):
        initializer.init_weights(model="not_a_model")


def test_init_weights_composite_recursion(initializer, mock_config):
    """Test that CompositeModel recursively calls init_weights on submodels."""
    composite = MagicMock(spec=CompositeModel)
    composite.name = "composite"
    sub1 = MagicMock(spec=Model)
    sub2 = MagicMock(spec=Model)
    composite.submodels = {"part_a": sub1, "part_b": sub2}

    # Mock the internal _init_weights to verify recursion stops at Model
    # We mock init_weights on the instance itself to track recursive calls:
    with patch.object(initializer, "_init_weights") as mock_inner_init:
        initializer.init_weights(composite)

        # Verify _init_weights was called for submodels:
        assert mock_inner_init.call_count == 2
        mock_inner_init.assert_has_calls(
            [call(model=sub1, model_name="composite.part_a"), call(model=sub2, model_name="composite.part_b")],
            any_order=True,
        )


@patch(f"{MODULE_PATH}.compute_model_norm")
def test_keys_removal(mock_norm, initializer, mock_model):
    """Test removing specific keys from the loaded state dict."""
    initializer.keys_to_remove = ["bad_key"]
    loaded_state_dict = {"good_key": 1, "bad_key": 2}

    with patch.object(initializer, "_get_model_state_dict", return_value=(loaded_state_dict, "name", "path")):
        # Mock norm to change so the check passes:
        mock_norm.side_effect = [10.0, 20.0]  # Before, After

        initializer.init_weights(mock_model)

        # Verify load_state_dict called without "bad_key":
        args, _ = mock_model.load_state_dict.call_args
        passed_sd = args[0]
        assert "bad_key" not in passed_sd
        assert "good_key" in passed_sd


@patch(f"{MODULE_PATH}.compute_model_norm")
def test_patterns_removal(mock_norm, initializer, mock_model):
    """Test removing keys matching a substring pattern."""
    initializer.patterns_to_remove = ["head"]
    loaded_state_dict = {"backbone.conv": 1, "head.fc": 2, "head.bias": 3}

    with patch.object(initializer, "_get_model_state_dict", return_value=(loaded_state_dict, "name", "path")):
        mock_norm.side_effect = [10.0, 20.0]

        initializer.init_weights(mock_model)

        args, _ = mock_model.load_state_dict.call_args
        passed_sd = args[0]
        assert "backbone.conv" in passed_sd
        assert "head.fc" not in passed_sd
        assert "head.bias" not in passed_sd


@patch(f"{MODULE_PATH}.compute_model_norm")
def test_patterns_renaming(mock_norm, initializer, mock_model):
    """Test renaming keys based on src/dst patterns."""
    initializer.patterns_to_rename = [{"src": "backbone.", "dst": "encoder."}]
    loaded_state_dict = {"backbone.layer1": 1, "head.fc": 2}

    with patch.object(initializer, "_get_model_state_dict", return_value=(loaded_state_dict, "name", "path")):
        mock_norm.side_effect = [10.0, 20.0]

        initializer.init_weights(mock_model)

        args, _ = mock_model.load_state_dict.call_args
        passed_sd = args[0]

        # "backbone.layer1" should be renamed to "encoder.layer1":
        assert "encoder.layer1" in passed_sd
        assert "backbone.layer1" not in passed_sd
        # Unaffected key remains:
        assert "head.fc" in passed_sd


@patch(f"{MODULE_PATH}.compute_model_norm")
def test_patterns_instantiate(mock_norm, initializer, mock_model):
    """
    Test 'instantiating' patterns.
    This means ignoring the loaded value and keeping the current model's initialized value.
    """
    initializer.patterns_to_instantiate = ["fresh_layer"]

    loaded_val = torch.tensor([1.0])
    fresh_val = torch.tensor([2.0])
    old_val = torch.tensor([3.0])

    loaded_state_dict = {"fresh_layer.weight": loaded_val, "old_layer.weight": old_val}

    # The current model has a different value:
    current_state_dict = {"fresh_layer.weight": fresh_val}
    mock_model.state_dict.return_value = current_state_dict

    with patch.object(initializer, "_get_model_state_dict", return_value=(loaded_state_dict, "name", "path")):
        mock_norm.side_effect = [10.0, 20.0]

        initializer.init_weights(mock_model)

        args, _ = mock_model.load_state_dict.call_args
        passed_sd = args[0]

        assert torch.equal(passed_sd["fresh_layer.weight"], fresh_val)
        assert torch.equal(passed_sd["old_layer.weight"], old_val)


@patch(f"{MODULE_PATH}.compute_model_norm")
def test_raises_if_model_unchanged(mock_norm, initializer, mock_model):
    """Test that RuntimeError is raised if weights didn't change (silent failure check)."""
    loaded_state_dict = {"w": 1}

    with patch.object(initializer, "_get_model_state_dict", return_value=(loaded_state_dict, "name", "path")):
        # Norm before == Norm after:
        mock_norm.side_effect = [10.0, 10.0]

        with pytest.raises(RuntimeError, match="Model has not been properly initialized"):
            initializer.init_weights(mock_model)
