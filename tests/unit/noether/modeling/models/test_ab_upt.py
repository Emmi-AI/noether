#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from unittest.mock import patch

import pytest
import torch
from torch import nn

from noether.core.schemas.dataset import DomainDataSpec, FieldDimSpec, ModelDataSpecs
from noether.core.schemas.models import AnchorBranchedUPTConfig
from noether.core.schemas.modules.attention import TokenSpec
from noether.core.schemas.modules.blocks import TransformerBlockConfig
from noether.core.schemas.modules.encoders import SupernodePoolingConfig
from noether.modeling.models.ab_upt import AnchoredBranchedUPT

_MODULE_PATH = "noether.modeling.models.ab_upt"


class FakePerceiverBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        # Return 'q' if provided (Perceiver logic), else first arg
        return kwargs.get("q", args[0] if args else None), None


class FakeTransformerBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x, None


class FakeGenericModule(nn.Module):
    """Generic replacement for other modules."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


@pytest.fixture
def real_config():
    data_specs = ModelDataSpecs(
        position_dim=3,
        conditioning_dims=FieldDimSpec({"mach_number": 1, "alpha": 1}),
        domains={
            "surface": DomainDataSpec(
                output_dims=FieldDimSpec({"pressure": 1, "shear_stress": 3}),
                feature_dim=FieldDimSpec({"area": 1}),
            ),
            "volume": DomainDataSpec(
                output_dims=FieldDimSpec({"velocity": 3, "pressure": 1}),
                feature_dim=FieldDimSpec({"sdf": 1}),
            ),
        },
    )

    tf_config = TransformerBlockConfig(
        hidden_dim=64,
        num_heads=4,
        mlp_expansion_factor=4.0,
        use_bias=True,
        use_rope=True,
        dropout=0.0,
    )

    pool_config = SupernodePoolingConfig(
        hidden_dim=64,
        input_dim=3,
        radius=0.1,
        k=None,
    )

    return AnchorBranchedUPTConfig(
        kind="AnchoredBranchedUPT",
        name="ab_upt_test",
        hidden_dim=64,
        geometry_depth=2,
        physics_blocks=["perceiver", "self"],
        num_domain_decoder_blocks={"surface": 2, "volume": 2},
        transformer_block_config=tf_config,
        supernode_pooling_config=pool_config,
        init_weights="truncnormal002",
        data_specs=data_specs,
    )


@pytest.fixture
def model(real_config):
    """
    Instantiates the model with FAKE classes replacing the real blocks.
    Using `yield` keeps the patches active during the test execution.
    """
    # PATCHES MUST STAY ALIVE during the test so isinstance() checks inside the model work:
    with (
        patch(_MODULE_PATH + ".RopeFrequency", new=FakeGenericModule),
        patch(_MODULE_PATH + ".SupernodePooling", new=FakeGenericModule),
        patch(_MODULE_PATH + ".TransformerBlock", new=FakeTransformerBlock),
        patch(_MODULE_PATH + ".ContinuousSincosEmbed", new=FakeGenericModule),
        patch(_MODULE_PATH + ".MLP", new=FakeGenericModule),
        patch(_MODULE_PATH + ".PerceiverBlock", new=FakePerceiverBlock),
        patch(_MODULE_PATH + ".LinearProjection", new=FakeGenericModule),
    ):
        model = AnchoredBranchedUPT(config=real_config)

        # Manually set decoders to real Linear layers for correct output shapes:
        model.domain_decoder_projections["surface"] = nn.Linear(
            64, real_config.data_specs.domains["surface"].output_dims.total_dim
        )
        model.domain_decoder_projections["volume"] = nn.Linear(
            64, real_config.data_specs.domains["volume"].output_dims.total_dim
        )

        yield model


@pytest.fixture
def three_domain_config():
    """Config with 3 domains: surface, volume, wake."""
    data_specs = ModelDataSpecs(
        position_dim=3,
        domains={
            "surface": DomainDataSpec(output_dims=FieldDimSpec({"pressure": 1})),
            "volume": DomainDataSpec(output_dims=FieldDimSpec({"velocity": 3})),
            "wake": DomainDataSpec(output_dims=FieldDimSpec({"turbulence": 2})),
        },
    )
    tf_config = TransformerBlockConfig(
        hidden_dim=64,
        num_heads=4,
        mlp_expansion_factor=4.0,
        use_bias=True,
        use_rope=True,
        dropout=0.0,
    )
    pool_config = SupernodePoolingConfig(hidden_dim=64, input_dim=3, radius=0.1, k=None)

    return AnchorBranchedUPTConfig(
        kind="AnchoredBranchedUPT",
        name="ab_upt_3domain",
        hidden_dim=64,
        geometry_depth=1,
        physics_blocks=["perceiver", "self", "cross"],
        num_domain_decoder_blocks={"surface": 1, "volume": 1, "wake": 1},
        transformer_block_config=tf_config,
        supernode_pooling_config=pool_config,
        init_weights="truncnormal002",
        data_specs=data_specs,
    )


@pytest.fixture
def three_domain_model(three_domain_config):
    with (
        patch(_MODULE_PATH + ".RopeFrequency", new=FakeGenericModule),
        patch(_MODULE_PATH + ".SupernodePooling", new=FakeGenericModule),
        patch(_MODULE_PATH + ".TransformerBlock", new=FakeTransformerBlock),
        patch(_MODULE_PATH + ".ContinuousSincosEmbed", new=FakeGenericModule),
        patch(_MODULE_PATH + ".MLP", new=FakeGenericModule),
        patch(_MODULE_PATH + ".PerceiverBlock", new=FakePerceiverBlock),
        patch(_MODULE_PATH + ".LinearProjection", new=FakeGenericModule),
    ):
        model = AnchoredBranchedUPT(config=three_domain_config)
        for name in model.domain_names:
            model.domain_decoder_projections[name] = nn.Linear(
                64, three_domain_config.data_specs.domains[name].output_dims.total_dim
            )
        yield model


class TestAnchoredBranchedUPT:
    def test_init(self, model, real_config):
        assert isinstance(model, AnchoredBranchedUPT)
        assert model.use_geometry_branch is True
        assert len(model.physics_blocks) == 2

    def test_prepare_condition(self, model):
        assert model._prepare_condition(None) is None
        assert model._prepare_condition({}) is None

        res = model._prepare_condition({"geometry": torch.randn(2, 1, 2)})
        assert res.shape == (2, 2)

        res = model._prepare_condition({"geometry": torch.randn(2, 1, 2), "inflow": torch.randn(2, 5)})
        assert res.shape == (2, 7)

    def test_create_all_token_specs(self, model):
        batch_size = 1
        surface_pos = torch.randn(batch_size, 10, 3)
        volume_pos = torch.randn(batch_size, 20, 3)

        specs, per_domain = model._create_all_token_specs(
            domain_anchor_positions={"surface": surface_pos, "volume": volume_pos},
            domain_query_positions={},
        )

        names = [s.name for s in specs]
        assert "surface_anchors" in names
        assert "volume_anchors" in names
        assert len(specs) == 2

        q_surface = torch.randn(batch_size, 5, 3)
        q_volume = torch.randn(batch_size, 5, 3)

        specs, per_domain = model._create_all_token_specs(
            domain_anchor_positions={"surface": surface_pos, "volume": volume_pos},
            domain_query_positions={"surface": q_surface, "volume": q_volume},
        )

        names = [s.name for s in specs]
        assert "surface_queries" in names
        assert "volume_queries" in names
        assert len(specs) == 4

    def test_forward_shape_integration(self, model, real_config):
        batch_size = 2
        num_geometry_nodes = 100
        num_surface_nodes = 50
        num_volume_nodes = 30

        geometry_pos = torch.randn(batch_size * num_geometry_nodes, 3)
        geometry_idx = torch.zeros(batch_size * num_geometry_nodes, dtype=torch.long)
        geometry_batch = torch.zeros(batch_size * num_geometry_nodes, dtype=torch.long)

        surface_anchors = torch.randn(batch_size, num_surface_nodes, 3)
        volume_anchors = torch.randn(batch_size, num_volume_nodes, 3)

        # Override Mock methods for this specific test
        # Note: We must override the methods on the INSTANCES attached to the model
        model.encoder.forward = lambda *a, **k: torch.randn(batch_size, 128, 64)
        model.pos_embed.forward = lambda x, *a, **k: torch.randn(x.shape[0], x.shape[1], 64)
        model.rope.forward = lambda *a, **k: torch.randn(batch_size, 2000, 16)

        predictions, kv_cache = model(
            geometry_position=geometry_pos,
            geometry_supernode_idx=geometry_idx,
            geometry_batch_idx=geometry_batch,
            domain_anchor_positions={"surface": surface_anchors, "volume": volume_anchors},
        )

        expected_surf_keys = {f"surface_{k}" for k in real_config.data_specs.domains["surface"].output_dims.keys()}
        expected_vol_keys = {f"volume_{k}" for k in real_config.data_specs.domains["volume"].output_dims.keys()}

        for k in expected_surf_keys:
            assert k in predictions
        for k in expected_vol_keys:
            assert k in predictions

        assert predictions["surface_pressure"].shape == (batch_size, num_surface_nodes, 1)
        assert predictions["volume_velocity"].shape == (batch_size, num_volume_nodes, 3)

    def test_forward_with_queries(self, model):
        batch_size = 1

        model.encoder.forward = lambda *a, **k: torch.randn(batch_size, 64, 64)
        model.pos_embed.forward = lambda x, *a, **k: torch.randn(x.shape[0], x.shape[1], 64)
        model.rope.forward = lambda *a, **k: torch.randn(batch_size, 2000, 16)

        predictions, _ = model(
            geometry_position=torch.randn(10, 3),
            geometry_supernode_idx=torch.zeros(10, dtype=torch.long),
            geometry_batch_idx=torch.zeros(10, dtype=torch.long),
            domain_anchor_positions={
                "surface": torch.randn(batch_size, 10, 3),
                "volume": torch.randn(batch_size, 10, 3),
            },
            domain_query_positions={"surface": torch.randn(batch_size, 5, 3)},
        )

        assert "query_surface_pressure" in predictions
        assert predictions["query_surface_pressure"].shape == (batch_size, 5, 1)

    def test_split_domain_tensors(self, model):
        batch_size = 1
        hidden_dim = 10
        x = torch.randn(batch_size, 8, hidden_dim)
        specs = [
            TokenSpec(name="surface_anchors", size=2),
            TokenSpec(name="surface_queries", size=2),
            TokenSpec(name="volume_anchors", size=2),
            TokenSpec(name="volume_queries", size=2),
        ]

        result = model._split_domain_tensors(x, specs)

        assert result["surface"].shape == (batch_size, 4, hidden_dim)
        assert result["volume"].shape == (batch_size, 4, hidden_dim)
        assert torch.allclose(result["surface"], x[:, 0:4])
        assert torch.allclose(result["volume"], x[:, 4:8])

    def test_forward_xor_anchors_and_cache(self, model):
        """Providing both anchors and kv_cache, or neither, must raise ValueError."""
        batch_size = 1
        model.encoder.forward = lambda *a, **k: torch.randn(batch_size, 64, 64)
        model.pos_embed.forward = lambda x, *a, **k: torch.randn(x.shape[0], x.shape[1], 64)
        model.rope.forward = lambda *a, **k: torch.randn(batch_size, 2000, 16)

        anchors_kwargs = dict(
            geometry_position=torch.randn(10, 3),
            geometry_supernode_idx=torch.zeros(10, dtype=torch.long),
            geometry_batch_idx=torch.zeros(10, dtype=torch.long),
            domain_anchor_positions={
                "surface": torch.randn(batch_size, 10, 3),
                "volume": torch.randn(batch_size, 10, 3),
            },
        )

        # Both anchors and cache → error
        fake_cache = {"physics": [], "surface": [], "volume": []}
        with pytest.raises(ValueError, match="not both"):
            model(**anchors_kwargs, kv_cache=fake_cache)

        # Neither anchors nor cache → error
        with pytest.raises(ValueError, match="not both"):
            model(domain_query_positions={"surface": torch.randn(batch_size, 5, 3)})


class TestThreeDomainABUPT:
    """Tests for 3+ domain generalization (surface, volume, wake)."""

    def test_init_three_domains(self, three_domain_model):
        assert three_domain_model.domain_names == ["surface", "volume", "wake"]
        assert len(three_domain_model.domain_biases) == 3
        assert len(three_domain_model.domain_decoder_blocks) == 3
        assert len(three_domain_model.domain_decoder_projections) == 3

    def test_forward_three_domains(self, three_domain_model, three_domain_config):
        batch_size = 2
        num_surface, num_volume, num_wake = 40, 30, 20

        three_domain_model.encoder.forward = lambda *a, **k: torch.randn(batch_size, 64, 64)
        three_domain_model.pos_embed.forward = lambda x, *a, **k: torch.randn(x.shape[0], x.shape[1], 64)
        three_domain_model.rope.forward = lambda *a, **k: torch.randn(batch_size, 2000, 16)

        predictions, kv_cache = three_domain_model(
            geometry_position=torch.randn(batch_size * 100, 3),
            geometry_supernode_idx=torch.zeros(batch_size * 100, dtype=torch.long),
            geometry_batch_idx=torch.zeros(batch_size * 100, dtype=torch.long),
            domain_anchor_positions={
                "surface": torch.randn(batch_size, num_surface, 3),
                "volume": torch.randn(batch_size, num_volume, 3),
                "wake": torch.randn(batch_size, num_wake, 3),
            },
        )

        # Check all domain prediction keys are present with correct shapes
        assert predictions["surface_pressure"].shape == (batch_size, num_surface, 1)
        assert predictions["volume_velocity"].shape == (batch_size, num_volume, 3)
        assert predictions["wake_turbulence"].shape == (batch_size, num_wake, 2)

    def test_forward_three_domains_with_queries(self, three_domain_model):
        batch_size = 1

        three_domain_model.encoder.forward = lambda *a, **k: torch.randn(batch_size, 64, 64)
        three_domain_model.pos_embed.forward = lambda x, *a, **k: torch.randn(x.shape[0], x.shape[1], 64)
        three_domain_model.rope.forward = lambda *a, **k: torch.randn(batch_size, 2000, 16)

        predictions, _ = three_domain_model(
            geometry_position=torch.randn(10, 3),
            geometry_supernode_idx=torch.zeros(10, dtype=torch.long),
            geometry_batch_idx=torch.zeros(10, dtype=torch.long),
            domain_anchor_positions={
                "surface": torch.randn(batch_size, 10, 3),
                "volume": torch.randn(batch_size, 8, 3),
                "wake": torch.randn(batch_size, 6, 3),
            },
            domain_query_positions={
                "wake": torch.randn(batch_size, 5, 3),
            },
        )

        # Anchor predictions
        assert predictions["surface_pressure"].shape == (batch_size, 10, 1)
        assert predictions["volume_velocity"].shape == (batch_size, 8, 3)
        assert predictions["wake_turbulence"].shape == (batch_size, 6, 2)
        # Query predictions only for wake
        assert "query_wake_turbulence" in predictions
        assert predictions["query_wake_turbulence"].shape == (batch_size, 5, 2)
        assert "query_surface_pressure" not in predictions
