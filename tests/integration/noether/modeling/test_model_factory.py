#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import torch

from noether.core.factory import Factory
from noether.core.models import Model
from noether.core.schemas.dataset import AeroDataSpecs, FieldDimSpec
from noether.core.schemas.models import AnchorBranchedUPTConfig, TransformerConfig, UPTConfig
from noether.core.schemas.modules.blocks import PerceiverBlockConfig, TransformerBlockConfig
from noether.core.schemas.modules.decoders import DeepPerceiverDecoderConfig
from noether.core.schemas.modules.encoders import SupernodePoolingConfig
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


# Verify a minimal TransformerConfig instantiates via Factory and processes a tensor batch.
def test_model_factory_creates_transformer_and_runs_forward() -> None:
    config = TransformerConfig(
        kind="noether.modeling.models.transformer.Transformer",
        name="test_transformer",
        hidden_dim=8,
        num_heads=2,
        depth=2,
        mlp_expansion_factor=2,
        drop_path=0.0,
    )

    model = Factory().create(config)

    assert isinstance(model, Transformer)

    batch_size, seq_len = 2, 5
    sample = torch.randn(batch_size, seq_len, config.hidden_dim)
    output = model(sample, attn_kwargs={})

    # Check that the output has the expected shape
    assert output.shape == (batch_size, seq_len, config.hidden_dim)
    # Check that the output does not contain NaN values
    assert not torch.isnan(output).any()
    # Check that the output is finite
    assert torch.isfinite(output).all()
    # Check that the model has trainable parameters
    assert sum(param.requires_grad for param in model.parameters()) > 0


# Verify that the factory can create a UPT and run a forward pass without errors.
def test_model_factory_creates_upt_and_runs_forward() -> None:
    data_specs = AeroDataSpecs(
        position_dim=3,
        surface_output_dims=FieldDimSpec({"pressure": 1}),
        volume_output_dims=FieldDimSpec({"density": 1}),
    )

    config = UPTConfig(
        kind="noether.modeling.models.upt.UPT",
        name="test_upt",
        num_heads=2,
        hidden_dim=8,
        mlp_expansion_factor=2,
        approximator_depth=1,
        use_rope=False,
        supernode_pooling_config=SupernodePoolingConfig(
            hidden_dim=8,  # Match the transformer's hidden_dim
            input_dim=3,  # 3 for 3D positions
            k=3,  # Number of neighbors for kNN in supernode pooling
        ),
        approximator_config=TransformerBlockConfig(
            hidden_dim=8,
            num_heads=2,
            mlp_expansion_factor=2,
        ),
        decoder_config=DeepPerceiverDecoderConfig(
            perceiver_block_config=PerceiverBlockConfig(
                hidden_dim=8,
                num_heads=2,
                mlp_expansion_factor=2,
            ),
            depth=1,
            input_dim=3,
        ),
        bias_layers=False,
        data_specs=data_specs,
    )

    model = Factory().create(config)

    assert isinstance(model, UPT)

    batch_size = 2
    surface_points_per_sample = 6
    supernodes_per_sample = 3
    total_surface_points = batch_size * surface_points_per_sample

    surface_position = torch.randn(total_surface_points, data_specs.position_dim)
    # Create batch indices that repeat for each surface point in a sample
    surface_position_batch_idx = torch.arange(batch_size).repeat_interleave(surface_points_per_sample)
    # Arbitrarily choose first `n=supernodes_per_sample` points as supernodes for each sample
    # e.g. for batch_size=2, surface_points_per_sample=6, supernodes_per_sample=3, we would have:
    # surface_position_supernode_idx = [0, 1, 2, 6, 7, 8]
    surface_position_supernode_idx = torch.cat(
        [
            torch.arange(supernodes_per_sample) + sample_index * surface_points_per_sample
            for sample_index in range(batch_size)
        ]
    )

    query_tokens = 4
    query_position = torch.randn(batch_size, query_tokens, data_specs.position_dim)

    output = model(
        surface_position_batch_idx=surface_position_batch_idx,
        surface_position_supernode_idx=surface_position_supernode_idx,
        surface_position=surface_position,
        query_position=query_position,
    )

    # Check that the output has the expected shape
    assert output.shape == (batch_size, query_tokens, data_specs.total_output_dim)
    # Check that the output does not contain NaN values
    assert not torch.isnan(output).any()
    # Check that the output is finite
    assert torch.isfinite(output).all()
    # Check that the model has trainable parameters
    assert sum(param.requires_grad for param in model.parameters()) > 0


# Verify that the factory can create an AnchoredBranchedUPT and run a forward pass without errors.
def test_model_factory_creates_ab_upt_and_runs_forward() -> None:
    data_specs = AeroDataSpecs(
        position_dim=3,
        surface_output_dims=FieldDimSpec({"cp": 1}),
        volume_output_dims=FieldDimSpec({"temperature": 1}),
    )

    config = AnchorBranchedUPTConfig(
        kind="noether.modeling.models.ab_upt.AnchoredBranchedUPT",
        name="test_ab_upt",
        geometry_depth=1,
        hidden_dim=12,
        physics_blocks=["shared"],
        num_surface_blocks=1,
        num_volume_blocks=1,
        data_specs=data_specs,
        supernode_pooling_config=SupernodePoolingConfig(
            hidden_dim=12,
            input_dim=3,
            k=3,
        ),
        transformer_block_config=TransformerBlockConfig(
            hidden_dim=12,
            num_heads=2,
            mlp_expansion_factor=2,
            use_rope=True,
        ),
    )

    model = Factory().create(config)

    assert isinstance(model, AnchoredBranchedUPT)

    batch_size = 2
    geometry_points_per_sample = 6
    geometry_supernodes_per_sample = 3
    total_geometry_points = batch_size * geometry_points_per_sample

    geometry_position = torch.randn(total_geometry_points, data_specs.position_dim)
    # Create batch indices that repeat for each geometry point in a sample
    geometry_batch_idx = torch.arange(batch_size).repeat_interleave(geometry_points_per_sample)
    # Arbitrarily choose first `n=geometry_supernodes_per_sample` points as supernodes for each sample
    # e.g. for batch_size=2, surface_points_per_sample=6, supernodes_per_sample=3, we would have:
    # surface_position_supernode_idx = [0, 1, 2, 6, 7, 8]
    geometry_supernode_idx = torch.cat(
        [
            torch.arange(geometry_supernodes_per_sample) + sample_index * geometry_points_per_sample
            for sample_index in range(batch_size)
        ]
    )

    surface_anchor_tokens = 4
    volume_anchor_tokens = 3

    # Randomly choose surface and volume anchor positions for each sample
    surface_anchor_position = torch.randn(batch_size, surface_anchor_tokens, data_specs.position_dim)
    volume_anchor_position = torch.randn(batch_size, volume_anchor_tokens, data_specs.position_dim)

    predictions = model(
        geometry_position=geometry_position,
        geometry_supernode_idx=geometry_supernode_idx,
        geometry_batch_idx=geometry_batch_idx,
        surface_anchor_position=surface_anchor_position,
        volume_anchor_position=volume_anchor_position,
        query_surface_position=surface_anchor_position,  # Explicitly reusing anchor positions as query positions
        query_volume_position=volume_anchor_position,  # Explicitly reusing anchor positions as query positions
    )

    # Extract expected output keys based on the data specs defined above
    expected_surface_keys = {f"surface_{name}" for name in data_specs.surface_output_dims.keys()}
    expected_volume_keys = {f"volume_{name}" for name in data_specs.volume_output_dims.keys()}

    # Check that the expected output keys are present in the predictions
    assert expected_surface_keys.issubset(predictions.keys())
    assert expected_volume_keys.issubset(predictions.keys())

    # For each expected surface output, check that the shape is correct and values are finite
    for key in expected_surface_keys:
        surface_dim = data_specs.surface_output_dims[key.removeprefix("surface_")]
        assert predictions[key].shape == (batch_size, surface_anchor_tokens, surface_dim)
        assert not torch.isnan(predictions[key]).any()
        assert torch.isfinite(predictions[key]).all()

    for key in expected_volume_keys:
        volume_dim = data_specs.volume_output_dims[key.removeprefix("volume_")]
        assert predictions[key].shape == (batch_size, volume_anchor_tokens, volume_dim)
        assert not torch.isnan(predictions[key]).any()
        assert torch.isfinite(predictions[key]).all()

    # Check that the model has trainable parameters
    assert sum(param.requires_grad for param in model.parameters()) > 0
