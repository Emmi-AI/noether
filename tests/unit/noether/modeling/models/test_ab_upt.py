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


class FakeUntiedTransformerBlock(nn.Module):
    """Fake drop-in for UntiedTransformerBlock. Matches (x, kv_cache) return shape."""

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


class LinearReadout(nn.Module):
    """Minimal stand-in for ``ReadoutLayer`` — projects to ``out_features`` and ignores ``condition``."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, condition=None):
        return self.linear(x)


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
        model.domain_decoder_projections["surface"] = LinearReadout(
            64, real_config.data_specs.domains["surface"].output_dims.total_dim
        )
        model.domain_decoder_projections["volume"] = LinearReadout(
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
            model.domain_decoder_projections[name] = LinearReadout(
                64, three_domain_config.data_specs.domains[name].output_dims.total_dim
            )
        yield model


class TestAnchoredBranchedUPT:
    def test_init(self, model, real_config):
        assert isinstance(model, AnchoredBranchedUPT)
        assert model.use_geometry_branch is True
        assert len(model.physics_blocks) == 2

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


@pytest.fixture
def untied_config():
    """Two-domain config using the three ``*_untied`` physics block variants."""
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
        name="ab_upt_untied_test",
        hidden_dim=64,
        geometry_depth=1,
        physics_blocks=["perceiver", "self_untied", "cross_untied", "joint_untied"],
        num_domain_decoder_blocks={"surface": 1, "volume": 1},
        transformer_block_config=tf_config,
        supernode_pooling_config=pool_config,
        init_weights="truncnormal002",
        data_specs=data_specs,
    )


@pytest.fixture
def untied_model(untied_config):
    """Model with every real block (including ``UntiedTransformerBlock``) replaced by a fake."""
    with (
        patch(_MODULE_PATH + ".RopeFrequency", new=FakeGenericModule),
        patch(_MODULE_PATH + ".SupernodePooling", new=FakeGenericModule),
        patch(_MODULE_PATH + ".TransformerBlock", new=FakeTransformerBlock),
        patch(_MODULE_PATH + ".UntiedTransformerBlock", new=FakeUntiedTransformerBlock),
        patch(_MODULE_PATH + ".ContinuousSincosEmbed", new=FakeGenericModule),
        patch(_MODULE_PATH + ".MLP", new=FakeGenericModule),
        patch(_MODULE_PATH + ".PerceiverBlock", new=FakePerceiverBlock),
        patch(_MODULE_PATH + ".LinearProjection", new=FakeGenericModule),
    ):
        model = AnchoredBranchedUPT(config=untied_config)
        model.domain_decoder_projections["surface"] = LinearReadout(
            64, untied_config.data_specs.domains["surface"].output_dims.total_dim
        )
        model.domain_decoder_projections["volume"] = LinearReadout(
            64, untied_config.data_specs.domains["volume"].output_dims.total_dim
        )
        yield model


@pytest.fixture
def untied_mixed_config():
    """Config with a mix of shared (``self``) and untied (``cross_untied``) physics blocks."""
    data_specs = ModelDataSpecs(
        position_dim=3,
        domains={
            "surface": DomainDataSpec(output_dims=FieldDimSpec({"pressure": 1})),
            "volume": DomainDataSpec(output_dims=FieldDimSpec({"velocity": 3})),
        },
    )
    tf_config = TransformerBlockConfig(
        hidden_dim=32,
        num_heads=4,
        mlp_expansion_factor=2.0,
        use_bias=True,
        use_rope=True,
        dropout=0.0,
    )
    pool_config = SupernodePoolingConfig(hidden_dim=32, input_dim=3, radius=0.1, k=None)

    return AnchorBranchedUPTConfig(
        kind="AnchoredBranchedUPT",
        name="ab_upt_mixed_untied",
        hidden_dim=32,
        geometry_depth=1,
        physics_blocks=["perceiver", "self", "cross_untied"],
        num_domain_decoder_blocks={"surface": 1, "volume": 1},
        transformer_block_config=tf_config,
        supernode_pooling_config=pool_config,
        init_weights="truncnormal002",
        data_specs=data_specs,
    )


@pytest.fixture
def untied_mixed_model(untied_mixed_config):
    with (
        patch(_MODULE_PATH + ".RopeFrequency", new=FakeGenericModule),
        patch(_MODULE_PATH + ".SupernodePooling", new=FakeGenericModule),
        patch(_MODULE_PATH + ".TransformerBlock", new=FakeTransformerBlock),
        patch(_MODULE_PATH + ".UntiedTransformerBlock", new=FakeUntiedTransformerBlock),
        patch(_MODULE_PATH + ".ContinuousSincosEmbed", new=FakeGenericModule),
        patch(_MODULE_PATH + ".MLP", new=FakeGenericModule),
        patch(_MODULE_PATH + ".PerceiverBlock", new=FakePerceiverBlock),
        patch(_MODULE_PATH + ".LinearProjection", new=FakeGenericModule),
    ):
        model = AnchoredBranchedUPT(config=untied_mixed_config)
        for name in model.domain_names:
            model.domain_decoder_projections[name] = LinearReadout(
                32, untied_mixed_config.data_specs.domains[name].output_dims.total_dim
            )
        yield model


class TestAnchoredBranchedUPTUntied:
    """Tests exercising the ``*_untied`` physics block variants.

    ``FakeUntiedTransformerBlock`` stands in for the real module so these tests
    focus on the model's wiring (suffix parsing, block dispatch, forward shapes)
    rather than the untied block's internal mechanics.
    """

    def test_init_dispatches_to_untied_blocks(self, untied_model):
        """``*_untied`` suffix builds ``UntiedTransformerBlock`` for every attention variant."""
        # perceiver is unaffected; the three untied entries each produce an untied block.
        assert len(untied_model.physics_blocks) == 4
        assert isinstance(untied_model.physics_blocks[0], FakePerceiverBlock)
        assert all(isinstance(b, FakeUntiedTransformerBlock) for b in untied_model.physics_blocks[1:]), (
            "self_untied / cross_untied / joint_untied should all instantiate UntiedTransformerBlock"
        )
        # No shared TransformerBlock should be constructed for the untied physics entries.
        physics_tf = [b for b in untied_model.physics_blocks if isinstance(b, FakeTransformerBlock)]
        assert physics_tf == []

    def test_forward_shape_with_untied_blocks(self, untied_model, untied_config):
        """Forward pass through untied physics blocks produces the same prediction keys/shapes."""
        batch_size = 2
        num_surface, num_volume = 20, 15

        untied_model.encoder.forward = lambda *a, **k: torch.randn(batch_size, 128, 64)
        untied_model.pos_embed.forward = lambda x, *a, **k: torch.randn(x.shape[0], x.shape[1], 64)
        untied_model.rope.forward = lambda *a, **k: torch.randn(batch_size, 2000, 16)

        predictions, _ = untied_model(
            geometry_position=torch.randn(50, 3),
            geometry_supernode_idx=torch.zeros(50, dtype=torch.long),
            geometry_batch_idx=torch.zeros(50, dtype=torch.long),
            domain_anchor_positions={
                "surface": torch.randn(batch_size, num_surface, 3),
                "volume": torch.randn(batch_size, num_volume, 3),
            },
        )

        expected_keys = set()
        for name, spec in untied_config.data_specs.domains.items():
            expected_keys.update(f"{name}_{field}" for field in spec.output_dims.keys())
        assert expected_keys.issubset(predictions.keys())

        assert predictions["surface_pressure"].shape == (batch_size, num_surface, 1)
        assert predictions["volume_velocity"].shape == (batch_size, num_volume, 3)

    def test_forward_with_queries_untied(self, untied_model):
        """Queries route through untied physics blocks and emerge under ``query_*`` keys."""
        batch_size = 1

        untied_model.encoder.forward = lambda *a, **k: torch.randn(batch_size, 64, 64)
        untied_model.pos_embed.forward = lambda x, *a, **k: torch.randn(x.shape[0], x.shape[1], 64)
        untied_model.rope.forward = lambda *a, **k: torch.randn(batch_size, 2000, 16)

        predictions, _ = untied_model(
            geometry_position=torch.randn(10, 3),
            geometry_supernode_idx=torch.zeros(10, dtype=torch.long),
            geometry_batch_idx=torch.zeros(10, dtype=torch.long),
            domain_anchor_positions={
                "surface": torch.randn(batch_size, 10, 3),
                "volume": torch.randn(batch_size, 10, 3),
            },
            domain_query_positions={"surface": torch.randn(batch_size, 5, 3)},
        )

        assert predictions["query_surface_pressure"].shape == (batch_size, 5, 1)
        # Volume has no queries → no query key for volume.
        assert "query_volume_velocity" not in predictions

    def test_untied_block_receives_correct_num_types(self, untied_config):
        """``num_types`` passed into ``UntiedTransformerBlockConfig`` equals the number of domains."""
        # Capture the configs that each UntiedTransformerBlock is constructed with.
        constructor_configs: list = []

        class RecordingFakeUntied(FakeUntiedTransformerBlock):
            def __init__(self, *args, **kwargs):
                super().__init__()
                constructor_configs.append(kwargs.get("config"))

        with (
            patch(_MODULE_PATH + ".RopeFrequency", new=FakeGenericModule),
            patch(_MODULE_PATH + ".SupernodePooling", new=FakeGenericModule),
            patch(_MODULE_PATH + ".TransformerBlock", new=FakeTransformerBlock),
            patch(_MODULE_PATH + ".UntiedTransformerBlock", new=RecordingFakeUntied),
            patch(_MODULE_PATH + ".ContinuousSincosEmbed", new=FakeGenericModule),
            patch(_MODULE_PATH + ".MLP", new=FakeGenericModule),
            patch(_MODULE_PATH + ".PerceiverBlock", new=FakePerceiverBlock),
            patch(_MODULE_PATH + ".LinearProjection", new=FakeGenericModule),
        ):
            AnchoredBranchedUPT(config=untied_config)

        # 3 physics entries are *_untied → 3 UntiedTransformerBlock instantiations
        assert len(constructor_configs) == 3
        num_domains = len(untied_config.data_specs.domains)  # surface + volume → 2
        for cfg in constructor_configs:
            assert cfg.num_types == num_domains
            # The nested TransformerBlockConfig should be carried through intact.
            assert cfg.transformer_block.hidden_dim == untied_config.hidden_dim

    def test_untied_blocks_have_independent_parameters(self, untied_config):
        """Each ``UntiedTransformerBlock`` contributes its own distinct parameters.

        Uses the *real* UntiedTransformerBlock (via patches only on the non-attention
        periphery) so that parameter counting is meaningful.
        """
        # Patch peripheral modules only; leave TransformerBlock / UntiedTransformerBlock real.
        with (
            patch(_MODULE_PATH + ".RopeFrequency", new=FakeGenericModule),
            patch(_MODULE_PATH + ".SupernodePooling", new=FakeGenericModule),
            patch(_MODULE_PATH + ".ContinuousSincosEmbed", new=FakeGenericModule),
            patch(_MODULE_PATH + ".MLP", new=FakeGenericModule),
            patch(_MODULE_PATH + ".PerceiverBlock", new=FakePerceiverBlock),
            patch(_MODULE_PATH + ".LinearProjection", new=FakeGenericModule),
        ):
            model = AnchoredBranchedUPT(config=untied_config)

        # Each untied block's q/k/v weight banks hold one 2D Parameter per type;
        # disjoint parameter tensors across blocks means no accidental sharing.
        # For AB-UPT, ``attention_block`` is a MultiBranchAnchorAttention wrapper
        # (Self/Cross/JointAnchorAttention); its inner ``mixed_attention`` is the
        # UntiedMixedAttention that owns the per-type q/k/v weights.
        untied_blocks = [b for b in model.physics_blocks if b.__class__.__name__ == "UntiedTransformerBlock"]
        assert len(untied_blocks) == 3
        num_types = len(untied_config.data_specs.domains)
        for proj_name in ("q", "k", "v"):
            proj_ids = {id(getattr(b.attention_block.mixed_attention, proj_name).weight) for b in untied_blocks}
            assert len(proj_ids) == 3, f"each UntiedTransformerBlock must hold its own {proj_name} weight tensor"
            for b in untied_blocks:
                assert len(getattr(b.attention_block.mixed_attention, proj_name).weight) == num_types

    def test_untied_block_uses_configured_attention_constructor(self, untied_config):
        """``*_untied`` blocks must honor the configured ``attention_constructor``.

        Pins the fix for a bug where ``UntiedTransformerBlock`` previously
        replaced ``self.attention_block`` wholesale with ``UntiedMixedAttention``,
        silently discarding the per-branch attention pattern (``self`` / ``cross``
        / ``joint``). Post-fix, ``attention_block`` is the configured
        ``MultiBranchAnchorAttention`` subclass and only its inner
        ``mixed_attention`` is the untied workhorse — so ``self_untied`` really
        emits per-branch self-attention patterns, not joint attention.
        """
        from noether.modeling.modules.attention.anchor_attention import (
            CrossAnchorAttention,
            JointAnchorAttention,
            SelfAnchorAttention,
        )
        from noether.modeling.modules.untied import UntiedMixedAttention

        with (
            patch(_MODULE_PATH + ".RopeFrequency", new=FakeGenericModule),
            patch(_MODULE_PATH + ".SupernodePooling", new=FakeGenericModule),
            patch(_MODULE_PATH + ".ContinuousSincosEmbed", new=FakeGenericModule),
            patch(_MODULE_PATH + ".MLP", new=FakeGenericModule),
            patch(_MODULE_PATH + ".PerceiverBlock", new=FakePerceiverBlock),
            patch(_MODULE_PATH + ".LinearProjection", new=FakeGenericModule),
        ):
            model = AnchoredBranchedUPT(config=untied_config)

        expected = {
            "self_untied": SelfAnchorAttention,
            "cross_untied": CrossAnchorAttention,
            "joint_untied": JointAnchorAttention,
        }
        # untied_config.physics_blocks is ["perceiver", "self_untied", "cross_untied", "joint_untied"].
        for block, block_type in zip(
            model.physics_blocks[1:], ["self_untied", "cross_untied", "joint_untied"], strict=True
        ):
            assert isinstance(block.attention_block, expected[block_type]), (
                f"{block_type} should build {expected[block_type].__name__}, got {type(block.attention_block).__name__}"
            )
            # The inner mixed_attention is the per-type workhorse.
            assert isinstance(block.attention_block.mixed_attention, UntiedMixedAttention)


class TestMixedTiedAndUntiedBlocks:
    """Models can mix shared (``self``) and untied (``cross_untied``) blocks in one list."""

    def test_init_mixed_blocks(self, untied_mixed_model):
        """Mixed physics_blocks yields the expected concrete block types in order."""
        blocks = untied_mixed_model.physics_blocks
        assert len(blocks) == 3
        assert isinstance(blocks[0], FakePerceiverBlock)
        assert isinstance(blocks[1], FakeTransformerBlock)  # "self" → shared TransformerBlock
        assert isinstance(blocks[2], FakeUntiedTransformerBlock)  # "cross_untied" → untied

    def test_forward_mixed_blocks(self, untied_mixed_model):
        """Mixing tied and untied physics blocks runs end-to-end without errors."""
        batch_size = 1

        untied_mixed_model.encoder.forward = lambda *a, **k: torch.randn(batch_size, 64, 32)
        untied_mixed_model.pos_embed.forward = lambda x, *a, **k: torch.randn(x.shape[0], x.shape[1], 32)
        untied_mixed_model.rope.forward = lambda *a, **k: torch.randn(batch_size, 1000, 8)

        predictions, _ = untied_mixed_model(
            geometry_position=torch.randn(10, 3),
            geometry_supernode_idx=torch.zeros(10, dtype=torch.long),
            geometry_batch_idx=torch.zeros(10, dtype=torch.long),
            domain_anchor_positions={
                "surface": torch.randn(batch_size, 8, 3),
                "volume": torch.randn(batch_size, 6, 3),
            },
        )

        assert predictions["surface_pressure"].shape == (batch_size, 8, 1)
        assert predictions["volume_velocity"].shape == (batch_size, 6, 3)


@pytest.fixture
def partial_feature_config():
    """Config where only one domain (``surface``) declares ``feature_dim``."""
    data_specs = ModelDataSpecs(
        position_dim=3,
        domains={
            "surface": DomainDataSpec(
                output_dims=FieldDimSpec({"pressure": 1}),
                feature_dim=FieldDimSpec({"area": 1}),
            ),
            "volume": DomainDataSpec(
                output_dims=FieldDimSpec({"velocity": 3}),
            ),
        },
    )
    tf_config = TransformerBlockConfig(
        hidden_dim=32,
        num_heads=4,
        mlp_expansion_factor=2.0,
        use_bias=True,
        use_rope=True,
        dropout=0.0,
    )
    pool_config = SupernodePoolingConfig(hidden_dim=32, input_dim=3, radius=0.1, k=None)

    return AnchorBranchedUPTConfig(
        kind="AnchoredBranchedUPT",
        name="ab_upt_partial_feat",
        hidden_dim=32,
        geometry_depth=1,
        physics_blocks=["perceiver", "self"],
        num_domain_decoder_blocks={"surface": 1, "volume": 1},
        transformer_block_config=tf_config,
        supernode_pooling_config=pool_config,
        init_weights="truncnormal002",
        data_specs=data_specs,
    )


@pytest.fixture
def partial_feature_model(partial_feature_config):
    with (
        patch(_MODULE_PATH + ".RopeFrequency", new=FakeGenericModule),
        patch(_MODULE_PATH + ".SupernodePooling", new=FakeGenericModule),
        patch(_MODULE_PATH + ".TransformerBlock", new=FakeTransformerBlock),
        patch(_MODULE_PATH + ".ContinuousSincosEmbed", new=FakeGenericModule),
        patch(_MODULE_PATH + ".MLP", new=FakeGenericModule),
        patch(_MODULE_PATH + ".PerceiverBlock", new=FakePerceiverBlock),
        patch(_MODULE_PATH + ".LinearProjection", new=FakeGenericModule),
    ):
        model = AnchoredBranchedUPT(config=partial_feature_config)
        for name in model.domain_names:
            model.domain_decoder_projections[name] = nn.Linear(
                32, partial_feature_config.data_specs.domains[name].output_dims.total_dim
            )
        yield model


class TestAnchoredBranchedUPTFeatures:
    """Tests for per-domain anchor/query input features.

    Per-domain feature MLPs are built only when
    ``data_specs.domains[name].feature_dim`` is set; the projected features
    are added to the position-bias output inside ``build_physics_input``.
    These tests cover the wiring (which projections get built), forward
    shapes when features are passed, and that features actually contribute
    to the physics input.
    """

    def test_feature_projs_built_for_each_configured_domain(self, model):
        """Both surface and volume declare ``feature_dim`` → projections built for both."""
        assert model.domain_feature_projs is not None
        assert set(model.domain_feature_projs.keys()) == {"surface", "volume"}

    def test_feature_projs_none_when_no_domain_has_feature_dim(self, three_domain_model):
        """No domain declares ``feature_dim`` → ``domain_feature_projs`` stays ``None``."""
        assert three_domain_model.domain_feature_projs is None

    def test_feature_projs_only_for_configured_domains(self, partial_feature_model):
        """Only ``surface`` declares ``feature_dim`` → only ``surface`` gets a projection."""
        assert partial_feature_model.domain_feature_projs is not None
        assert "surface" in partial_feature_model.domain_feature_projs
        assert "volume" not in partial_feature_model.domain_feature_projs

    def test_forward_with_anchor_features(self, model):
        """Anchor features for every configured domain run end-to-end with correct shapes."""
        batch_size = 2
        num_surface, num_volume = 20, 15

        model.encoder.forward = lambda *a, **k: torch.randn(batch_size, 64, 64)
        model.pos_embed.forward = lambda x, *a, **k: torch.randn(x.shape[0], x.shape[1], 64)
        model.rope.forward = lambda *a, **k: torch.randn(batch_size, 2000, 16)

        predictions, _ = model(
            geometry_position=torch.randn(50, 3),
            geometry_supernode_idx=torch.zeros(50, dtype=torch.long),
            geometry_batch_idx=torch.zeros(50, dtype=torch.long),
            domain_anchor_positions={
                "surface": torch.randn(batch_size, num_surface, 3),
                "volume": torch.randn(batch_size, num_volume, 3),
            },
            domain_anchor_features={
                "surface": torch.randn(batch_size, num_surface, 1),  # area
                "volume": torch.randn(batch_size, num_volume, 1),  # sdf
            },
        )

        assert predictions["surface_pressure"].shape == (batch_size, num_surface, 1)
        assert predictions["volume_velocity"].shape == (batch_size, num_volume, 3)

    def test_forward_with_anchor_and_query_features(self, model):
        """Anchor and query features coexist; query side emerges under ``query_*`` keys."""
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
                "volume": torch.randn(batch_size, 8, 3),
            },
            domain_query_positions={
                "surface": torch.randn(batch_size, 5, 3),
            },
            domain_anchor_features={
                "surface": torch.randn(batch_size, 10, 1),
                "volume": torch.randn(batch_size, 8, 1),
            },
            domain_query_features={
                "surface": torch.randn(batch_size, 5, 1),
            },
        )

        assert predictions["surface_pressure"].shape == (batch_size, 10, 1)
        assert predictions["volume_velocity"].shape == (batch_size, 8, 3)
        assert predictions["query_surface_pressure"].shape == (batch_size, 5, 1)

    def test_forward_with_query_features_only(self, model):
        """Query-only features (no anchor features supplied) still flow through."""
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
                "volume": torch.randn(batch_size, 8, 3),
            },
            domain_query_positions={
                "surface": torch.randn(batch_size, 5, 3),
            },
            domain_query_features={
                "surface": torch.randn(batch_size, 5, 1),
            },
        )

        assert predictions["query_surface_pressure"].shape == (batch_size, 5, 1)

    def test_features_contribute_to_physics_input(self, model):
        """Anchor features actually contribute to ``x_physics`` — output differs with vs. without."""
        batch_size = 1
        # Force pos_embed to a constant so the only source of variation is the feature path.
        model.pos_embed.forward = lambda x, *a, **k: torch.zeros(x.shape[0], x.shape[1], 64)

        anchor_pos = {
            "surface": torch.randn(batch_size, 4, 3),
            "volume": torch.randn(batch_size, 3, 3),
        }
        anchor_feat = {
            "surface": torch.ones(batch_size, 4, 1),
            "volume": torch.ones(batch_size, 3, 1),
        }

        x_no_feat, _ = model.build_physics_input(domain_anchor_positions=anchor_pos)
        x_with_feat, _ = model.build_physics_input(
            domain_anchor_positions=anchor_pos,
            domain_anchor_features=anchor_feat,
        )

        assert x_no_feat.shape == x_with_feat.shape
        assert not torch.allclose(x_no_feat, x_with_feat)

    def test_query_features_contribute_to_physics_input(self, model):
        """Query features change the query segment of ``x_physics`` independently of anchors."""
        batch_size = 1
        model.pos_embed.forward = lambda x, *a, **k: torch.zeros(x.shape[0], x.shape[1], 64)

        anchor_pos = {
            "surface": torch.randn(batch_size, 4, 3),
            "volume": torch.randn(batch_size, 3, 3),
        }
        query_pos = {"surface": torch.randn(batch_size, 2, 3)}
        query_feat = {"surface": torch.ones(batch_size, 2, 1)}

        x_no_query_feat, _ = model.build_physics_input(
            domain_anchor_positions=anchor_pos,
            domain_query_positions=query_pos,
        )
        x_with_query_feat, _ = model.build_physics_input(
            domain_anchor_positions=anchor_pos,
            domain_query_positions=query_pos,
            domain_query_features=query_feat,
        )

        assert x_no_query_feat.shape == x_with_query_feat.shape
        # Anchor segment for surface is the first 4 tokens — must remain unchanged.
        assert torch.allclose(x_no_query_feat[:, :4], x_with_query_feat[:, :4])
        # Query segment for surface (tokens 4:6) must differ.
        assert not torch.allclose(x_no_query_feat[:, 4:6], x_with_query_feat[:, 4:6])

    def test_features_silently_ignored_without_projection(self, three_domain_model):
        """Features supplied for domains without ``feature_dim`` are dropped, not raised."""
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
            domain_anchor_features={
                "surface": torch.randn(batch_size, 10, 1),
                "volume": torch.randn(batch_size, 8, 1),
                "wake": torch.randn(batch_size, 6, 1),
            },
        )

        assert predictions["surface_pressure"].shape == (batch_size, 10, 1)
        assert predictions["wake_turbulence"].shape == (batch_size, 6, 2)

    def test_features_applied_only_for_configured_domain(self, partial_feature_model):
        """Mixed config: surface features change ``x_physics`` but volume features are ignored."""
        batch_size = 1
        partial_feature_model.pos_embed.forward = lambda x, *a, **k: torch.zeros(x.shape[0], x.shape[1], 32)

        anchor_pos = {
            "surface": torch.randn(batch_size, 4, 3),
            "volume": torch.randn(batch_size, 3, 3),
        }

        x_baseline, _ = partial_feature_model.build_physics_input(domain_anchor_positions=anchor_pos)
        x_volume_feat, _ = partial_feature_model.build_physics_input(
            domain_anchor_positions=anchor_pos,
            domain_anchor_features={"volume": torch.ones(batch_size, 3, 1)},
        )
        x_surface_feat, _ = partial_feature_model.build_physics_input(
            domain_anchor_positions=anchor_pos,
            domain_anchor_features={"surface": torch.ones(batch_size, 4, 1)},
        )

        # Volume has no projection → its features are dropped, output matches baseline.
        assert torch.allclose(x_baseline, x_volume_feat)
        # Surface has a projection → its features change the output.
        assert not torch.allclose(x_baseline, x_surface_feat)
