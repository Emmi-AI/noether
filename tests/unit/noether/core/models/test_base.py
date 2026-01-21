#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from noether.core.models.base import ModelBase
from noether.core.schemas.models.base import ModelBaseConfig

MODULE_PATH = "noether.core.models.base"


class StubModel(ModelBase):
    """
    A minimal concrete implementation of ModelBase.
    We implement the abstract methods as no-ops or mocks so we can
    test the concrete logic residing in ModelBase itself.
    """

    def __init__(self, config, **kwargs):
        super().__init__(model_config=config, **kwargs)
        self.layer_1 = nn.Linear(2, 2)
        self.layer_2 = nn.Linear(2, 2)

        # Freeze layer_2 manually to test trainable/frozen counts:
        for param in self.layer_2.parameters():
            param.requires_grad = False

    # --- Implement Abstract Methods:
    def get_named_models(self):
        return {"stub": self}

    def initialize_weights(self):
        return self

    def apply_initializers(self):
        return self

    def initialize_optimizer(self):
        pass

    def optimizer_step(self, grad_scaler):
        pass

    def optimizer_schedule_step(self):
        pass

    def optimizer_zero_grad(self, set_to_none=True):
        pass

    # --- Implement Abstract Properties:
    @property
    def device(self):
        return torch.device("cpu")

    @property
    def is_frozen(self):
        return False


@pytest.fixture
def model_base_config():
    return ModelBaseConfig(kind="stub", name="stub_model")


@patch(f"{MODULE_PATH}.Factory")  # Patch factory to avoid loading real Initializers
def test_initialization_orchestration(MockFactory, model_base_config):
    """
    Test that the base 'initialize()' method correctly calls the
    lifecycle methods in order and sets the state flag.
    """
    model = StubModel(model_base_config)

    model.initialize_weights = MagicMock()
    model.initialize_optimizer = MagicMock()
    model.apply_initializers = MagicMock()

    assert model.is_initialized is False

    model.initialize()

    assert model.is_initialized is True
    model.initialize_weights.assert_called_once()
    model.initialize_optimizer.assert_called_once()
    model.apply_initializers.assert_called_once()


@patch(f"{MODULE_PATH}.Factory")
def test_parameter_counts(MockFactory, model_base_config):
    """Test the parameter counting properties implementation in ModelBase."""
    model = StubModel(model_base_config)

    # layer_1 (2x2 weight + 2 bias) = 6 params (Trainable)
    # layer_2 (2x2 weight + 2 bias) = 6 params (Frozen)

    assert model.param_count == 12
    assert model.trainable_param_count == 6
    assert model.frozen_param_count == 6


@patch(f"{MODULE_PATH}.Factory")
def test_nograd_paramnames(MockFactory, model_base_config):
    """
    Test detection of parameters that require grad but have no grad attribute.
    This usually happens before the first backward pass.
    """
    model = StubModel(model_base_config)

    # layer_1 requires grad, but we haven't run backward(), so .grad is None. They should appear in this list.
    nograd_names = model.nograd_paramnames

    assert "layer_1.weight" in nograd_names
    assert "layer_1.bias" in nograd_names

    # layer_2 does NOT require grad, so it should NOT be in this list.
    assert "layer_2.weight" not in nograd_names

    # Now simulate a gradient on layer_1.weight:
    model.layer_1.weight.grad = torch.ones_like(model.layer_1.weight)

    # Should vanish from the list:
    assert "layer_1.weight" not in model.nograd_paramnames
    # Bias still has no grad:
    assert "layer_1.bias" in model.nograd_paramnames
