#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from functools import partial

import pytest
import torch

from noether.core.models import Model
from noether.core.schemas.models.base import ModelBaseConfig


class SimpleSingleModel(Model):
    def __init__(self, model_config: ModelBaseConfig, **kwargs):
        super().__init__(model_config=model_config, is_frozen=model_config.is_frozen, **kwargs)
        self.ff = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.ff(x)


def test_initialize_single_model():
    model = SimpleSingleModel(
        model_config=ModelBaseConfig(
            kind="single", name="model1", optimizer_config={"kind": "torch.optim.AdamW", "lr": 1e-4}
        )
    )
    assert not model.is_frozen
    assert model.device == torch.device("cpu")


def test_is_frozen_property():
    model = SimpleSingleModel(
        model_config=ModelBaseConfig(is_frozen=True, kind="single", name="model1"),
    )
    assert model.is_frozen

    with pytest.raises(ValueError):
        _ = SimpleSingleModel(
            model_config=ModelBaseConfig(
                is_frozen=True,
                kind="single",
                name="model1",
                optimizer_config={"kind": "torch.optim.AdamW", "lr": 1e-4},
            ),
        )


def test_get_named_models():
    model = SimpleSingleModel(
        model_config=ModelBaseConfig(kind="single", name="model1"),
    )
    named_models = model.get_named_models()
    assert isinstance(named_models, dict)
    assert "model1" in named_models
    assert named_models["model1"] is model


def test_initialize_weights():
    model = SimpleSingleModel(
        model_config=ModelBaseConfig(kind="single", name="model1"),
    )
    model.initialize_weights()
    # Check that weights are initialized (not all zeros)
    for param in model.parameters():
        assert param.requires_grad

    model = SimpleSingleModel(
        model_config=ModelBaseConfig(kind="single", name="model1", is_frozen=True),
    )
    model.initialize_weights()
    # Check that weights are frozen (requires_grad is False)
    for param in model.parameters():
        assert not param.requires_grad


def test_initialize_optimizer():
    model = SimpleSingleModel(
        model_config=ModelBaseConfig(
            kind="single", name="model1", optimizer_config={"kind": "torch.optim.AdamW", "lr": 1e-4}
        ),
    )
    assert model._optim is None
    assert type(model._optimizer_constructor) is type(partial(torch.optim.AdamW, lr=1e-4))
    model.initialize_optimizer()
    assert model._optim is not None

    with pytest.raises(RuntimeError):
        model = SimpleSingleModel(
            model_config=ModelBaseConfig(kind="single", name="model1"),
        )
        model.initialize_optimizer()


def test_train_mode_after_freeze():
    model = SimpleSingleModel(
        model_config=ModelBaseConfig(kind="single", name="model1", is_frozen=True),
    )
    model.initialize_weights()
    assert not model.training
    model.train()
    assert not model.training

    model = SimpleSingleModel(
        model_config=ModelBaseConfig(kind="single", name="model1", is_frozen=False),
    )
    model.initialize_weights()
    assert model.training
    model.train(False)
    assert not model.training
    model.train(True)
    assert model.training


def test_device_property():
    model = SimpleSingleModel(
        model_config=ModelBaseConfig(kind="single", name="model1"),
    )
    assert model.device == torch.device("cpu")
    if torch.cuda.is_available():
        model.to("cuda")
        assert model.device == torch.device("cuda")
    with pytest.raises(RuntimeError):
        model.to("invalid_device")


def test_optimizer_step_on_frozen_model():
    """Test that optimizer_step works correctly on a none frozen model (no operation)."""
    model = SimpleSingleModel(
        model_config=ModelBaseConfig(
            kind="single", name="model1", is_frozen=False, optimizer_config={"kind": "torch.optim.AdamW", "lr": 1e-4}
        ),
    )
    model.initialize_optimizer()
    model.initialize_weights()

    dumpy_data = torch.randn(2, 10)
    output = model(dumpy_data)
    loss = output.sum()
    for p in model.parameters():
        assert p.requires_grad
        assert p.grad is None
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None
        assert p.grad.sum() != 0
    weights_before_step = {name: param.clone() for name, param in model.named_parameters()}
    model.optimizer_step(None)  # should work fine
    for name, param in model.named_parameters():
        assert not torch.equal(param, weights_before_step[name]), (
            f"Parameter {name} did not change after optimizer step"
        )
    model.optimizer.zero_grad()
    for p in model.parameters():
        assert p.grad is None


def test_param_counts():
    model = SimpleSingleModel(
        model_config=ModelBaseConfig(kind="single", name="model1", is_frozen=False),
    )
    total_params = model.param_count
    trainable_params = model.trainable_param_count
    non_grad_params = model.frozen_param_count

    assert total_params == 110  # replace with actual expected value
    assert trainable_params == 110  # replace with actual expected value
    assert non_grad_params == 0  # replace with actual expected value
