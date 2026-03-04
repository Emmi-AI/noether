#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from noether.core.factory import Factory
from noether.core.models import Model
from noether.core.schemas.dataset import AeroDataSpecs
from noether.core.schemas.models import AnchorBranchedUPTConfig, TransformerConfig, UPTConfig
from noether.modeling.models.ab_upt import AnchoredBranchedUPT
from noether.modeling.models.transformer import Transformer
from noether.modeling.models.upt import UPT
from tests.test_training_pipeline.dummy_project.schemas.models.base_model_config import BaseModelConfig


# Verify that the factory can create a simple model and run a forward pass without errors.
def test_model_factory_initializes_model_and_runs_forward() -> None:
    config = BaseModelConfig(
        kind="tests.test_training_pipeline.dummy_project.models.base_model.BaseModel",
        name="test_model",
        input_dim=4,
        hidden_dim=8,
        output_dim=2,
        num_hidden_layers=1,
        dropout=0.1,
    )

    model = Factory().create(config)

    assert isinstance(model, Model)

    sample = torch.randn(3, config.input_dim)
    output = model(sample)

    # Check that the output has the expected shape
    assert output.shape == (3, config.output_dim)
    # Check that the output does not contain NaN values
    assert not torch.isnan(output).any()
    # Check that the output is finite
    assert torch.isfinite(output).all()
    # Check that the model has trainable parameters
    assert sum(param.requires_grad for param in model.parameters()) > 0


# Verify a minimal TransformerConfig instantiates via Factory, processes a tensor batch and parameters receive gradients.
def test_model_factory_creates_transformer_and_runs_forward(
    transformer_config: TransformerConfig,
) -> None:
    model = Factory().create(transformer_config)

    assert isinstance(model, Transformer)

    batch_size, seq_len = 2, 5
    sample = torch.randn(batch_size, seq_len, transformer_config.hidden_dim)
    output = model(sample, attn_kwargs={})

    # Check that the output has the expected shape
    assert output.shape == (batch_size, seq_len, transformer_config.hidden_dim)
    # Check that the output does not contain NaN values
    assert not torch.isnan(output).any()
    # Check that the output is finite
    assert torch.isfinite(output).all()
    # Check that the model has trainable parameters
    assert sum(param.requires_grad for param in model.parameters()) > 0

    # Check that all parameters receive gradients
    loss = output.sum()
    loss.backward()
    no_grad_params = [name for name, param in model.named_parameters() if param.requires_grad and param.grad is None]
    assert not no_grad_params, f"Parameters with no gradient: {no_grad_params}"


# Verify a minimal UPTConfig instantiates via Factory, processes a forward pass and parameters receive gradients.
def test_model_factory_creates_upt_and_runs_forward(
    upt_config: UPTConfig, upt_data_specs: AeroDataSpecs, upt_input_generator: Callable[[int | None], dict[str, Any]]
) -> None:
    model = Factory().create(upt_config)

    assert isinstance(model, UPT)

    # Generate inputs from pytest fixture
    inputs = upt_input_generator(seed=42)

    output = model(**inputs)

    batch_size = 2
    query_tokens = 4

    # Check that the output has the expected shape
    assert output.shape == (batch_size, query_tokens, upt_data_specs.total_output_dim)
    # Check that the output does not contain NaN values
    assert not torch.isnan(output).any()
    # Check that the output is finite
    assert torch.isfinite(output).all()
    # Check that the model has trainable parameters
    assert sum(param.requires_grad for param in model.parameters()) > 0

    # Check that parameters receive gradients
    loss = output.sum()
    loss.backward()
    no_grad_params = [name for name, param in model.named_parameters() if param.requires_grad and param.grad is None]
    assert not no_grad_params, f"Parameters with no gradient: {no_grad_params}"


# Verify a minimal AnchorBranchedUPTConfig instantiates via Factory, processes a forward pass and parameters receive gradients.
def test_model_factory_creates_ab_upt_and_runs_forward(
    ab_upt_config: AnchorBranchedUPTConfig,
    ab_upt_data_specs: AeroDataSpecs,
    ab_upt_input_generator: Callable[[int | None], dict[str, Any]],
) -> None:
    ab_upt_config.physics_blocks = ["perceiver", "shared"]
    model = Factory().create(ab_upt_config)

    assert isinstance(model, AnchoredBranchedUPT)

    # Generate inputs from pytest fixture
    inputs = ab_upt_input_generator(seed=42)
    predictions = model(**inputs)

    batch_size = 2
    surface_anchor_tokens = 4
    volume_anchor_tokens = 3

    # Extract expected output keys based on the data specs defined above
    expected_surface_keys = {f"surface_{name}" for name in ab_upt_data_specs.surface_output_dims.keys()}
    expected_volume_keys = {f"volume_{name}" for name in ab_upt_data_specs.volume_output_dims.keys()}

    # Check that the expected output keys are present in the predictions
    assert expected_surface_keys.issubset(predictions.keys())
    assert expected_volume_keys.issubset(predictions.keys())

    # For each expected surface output, check that the shape is correct and values are finite
    for key in expected_surface_keys:
        surface_dim = ab_upt_data_specs.surface_output_dims[key.removeprefix("surface_")]
        assert predictions[key].shape == (batch_size, surface_anchor_tokens, surface_dim)
        assert not torch.isnan(predictions[key]).any()
        assert torch.isfinite(predictions[key]).all()

    # For each expected volume output, check that the shape is correct and values are finite
    for key in expected_volume_keys:
        volume_dim = ab_upt_data_specs.volume_output_dims[key.removeprefix("volume_")]
        assert predictions[key].shape == (batch_size, volume_anchor_tokens, volume_dim)
        assert not torch.isnan(predictions[key]).any()
        assert torch.isfinite(predictions[key]).all()

    # Check that the model has trainable parameters
    assert sum(param.requires_grad for param in model.parameters()) > 0

    # Check that all parameters receive gradients
    loss = sum(tensor.sum() for tensor in predictions.values())
    loss.backward()
    no_grad_params = [name for name, param in model.named_parameters() if param.requires_grad and param.grad is None]
    assert not no_grad_params, f"Parameters with no gradient: {no_grad_params}"
