#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from noether.data.schemas import DomainDataSpec, FieldDimSpec, ModelDataSpecs
from noether.modeling.models.ab_upt import AnchorBranchedUPTConfig
from noether.modeling.models.transformer import TransformerConfig
from noether.modeling.models.transolver import TransolverConfig
from noether.modeling.models.upt import UPTConfig
from noether.modeling.modules.blocks.perceiver import PerceiverBlockConfig
from noether.modeling.modules.blocks.transformer import TransformerBlockConfig
from noether.modeling.modules.decoders.deep_perceiver import DeepPerceiverDecoderConfig
from noether.modeling.modules.encoders.supernode_pooling import SupernodePoolingConfig


@pytest.fixture
def transolver_config() -> TransolverConfig:
    return TransolverConfig(
        kind="noether.modeling.models.transolver.Transolver",
        name="test_transolver",
        hidden_dim=8,
        transformer_block_config={"num_heads": 2, "mlp_expansion_factor": 2, "drop_path": 0.0},
        depth=2,
    )


@pytest.fixture
def transformer_config() -> TransformerConfig:
    return TransformerConfig(
        kind="noether.modeling.models.transformer.Transformer",
        name="test_transformer",
        hidden_dim=8,
        transformer_block_config={"num_heads": 2, "mlp_expansion_factor": 2, "drop_path": 0.0},
        depth=2,
    )


@pytest.fixture
def upt_data_specs() -> ModelDataSpecs:
    return ModelDataSpecs(
        position_dim=3,
        domains={
            "surface": DomainDataSpec(output_dims=FieldDimSpec({"pressure": 1})),
            "volume": DomainDataSpec(output_dims=FieldDimSpec({"density": 1})),
        },
    )


@pytest.fixture
def upt_config(upt_data_specs: ModelDataSpecs) -> UPTConfig:
    return UPTConfig(
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
        data_specs=upt_data_specs,
    )


@pytest.fixture
def ab_upt_data_specs() -> ModelDataSpecs:
    return ModelDataSpecs(
        position_dim=3,
        domains={
            "surface": DomainDataSpec(output_dims=FieldDimSpec({"cp": 1})),
            "volume": DomainDataSpec(output_dims=FieldDimSpec({"temperature": 1})),
        },
    )


@pytest.fixture
def ab_upt_config(ab_upt_data_specs: ModelDataSpecs) -> AnchorBranchedUPTConfig:
    return AnchorBranchedUPTConfig(
        kind="noether.modeling.models.ab_upt.AnchoredBranchedUPT",
        name="test_ab_upt",
        geometry_depth=1,
        hidden_dim=12,
        physics_blocks=["perceiver", "self"],
        num_domain_decoder_blocks={"surface": 1, "volume": 1},
        data_specs=ab_upt_data_specs,
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


@pytest.fixture
def upt_input_generator(
    upt_data_specs: ModelDataSpecs,
) -> Callable[[int | None], dict[str, Any]]:
    def _generate(seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            gen = torch.Generator().manual_seed(seed)
        else:
            gen = None

        batch_size = 2
        surface_points_per_sample = 6
        supernodes_per_sample = 3
        total_surface_points = batch_size * surface_points_per_sample

        geometry_position = torch.randn(total_surface_points, upt_data_specs.position_dim, generator=gen)
        geometry_batch_idx = torch.arange(batch_size).repeat_interleave(surface_points_per_sample)
        geometry_supernode_idx = torch.cat(
            [
                torch.arange(supernodes_per_sample) + sample_index * surface_points_per_sample
                for sample_index in range(batch_size)
            ]
        )

        query_tokens = 4
        query_position = torch.randn(batch_size, query_tokens, upt_data_specs.position_dim, generator=gen)

        return {
            "geometry_position": geometry_position,
            "geometry_batch_idx": geometry_batch_idx,
            "geometry_supernode_idx": geometry_supernode_idx,
            "query_position": query_position,
        }

    return _generate


@pytest.fixture
def ab_upt_input_generator(
    ab_upt_data_specs: ModelDataSpecs,
) -> Callable[[int | None], dict[str, Any]]:
    def _generate(seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            gen = torch.Generator().manual_seed(seed)
        else:
            gen = None

        batch_size = 2
        geometry_points_per_sample = 6
        geometry_supernodes_per_sample = 3
        total_geometry_points = batch_size * geometry_points_per_sample

        geometry_position = torch.randn(total_geometry_points, ab_upt_data_specs.position_dim, generator=gen)
        # Create batch indices that repeat for each geometry point in a sample
        geometry_batch_idx = torch.arange(batch_size).repeat_interleave(geometry_points_per_sample)

        geometry_supernode_idx = torch.cat(
            [
                torch.arange(geometry_supernodes_per_sample) + sample_index * geometry_points_per_sample
                for sample_index in range(batch_size)
            ]
        )

        surface_anchor_tokens = 4
        volume_anchor_tokens = 3

        # Randomly choose surface and volume anchor positions for each sample
        surface_anchor_position = torch.randn(
            batch_size, surface_anchor_tokens, ab_upt_data_specs.position_dim, generator=gen
        )
        volume_anchor_position = torch.randn(
            batch_size, volume_anchor_tokens, ab_upt_data_specs.position_dim, generator=gen
        )

        return {
            "geometry_position": geometry_position,
            "geometry_supernode_idx": geometry_supernode_idx,
            "geometry_batch_idx": geometry_batch_idx,
            "domain_anchor_positions": {
                "surface": surface_anchor_position,
                "volume": volume_anchor_position,
            },
            "domain_query_positions": {
                "surface": surface_anchor_position,  # Explicitly reusing anchor positions as query positions
                "volume": volume_anchor_position,
            },
        }

    return _generate
