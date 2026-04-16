#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import math

import pytest
import torch
import torch.nn as nn

from noether.core.optimizer.param_group_modifiers.completep import CompletePModifier
from noether.core.schemas.models.transformer import TransformerConfig
from noether.core.schemas.modules.blocks import TransformerBlockConfig
from noether.core.schemas.optimizers import CompletePModifierConfig
from noether.core.schemas.parameterizations import CompletePConfig
from noether.modeling.models.transformer import Transformer
from noether.modeling.modules.blocks.transformer import TransformerBlock

# =====================================================================================================================
# CompletePConfig + TransformerConfig integration
# =====================================================================================================================


class TestCompletePConfig:
    def test_default_values(self):
        cfg = CompletePConfig()
        assert cfg.base_width == 256
        assert cfg.base_depth == 2
        assert cfg.depth_alpha_exp == 1.0
        assert cfg.init_std == 0.02
        assert cfg.output_alpha == 1.0

    def test_transformer_config_derives_values(self):
        """Test that TransformerConfig correctly computes derived CompleteP values."""
        config = TransformerConfig(
            name="test",
            hidden_dim=512,
            depth=8,
            transformer_block_config=TransformerBlockConfig(
                hidden_dim=512,
                num_heads=8,
                mlp_expansion_factor=4,
            ),
            completep_config=CompletePConfig(
                base_width=256,
                base_depth=2,
                depth_alpha_exp=1.0,
                init_std=0.02,
                output_alpha=1.0,
            ),
        )

        m_w = 512 / 256  # = 2.0
        m_d = 8 / 2  # = 4.0
        head_dim = 512 // 8  # = 64

        assert config.transformer_block_config.residual_scale == pytest.approx(1.0 / 4.0)
        assert config.transformer_block_config.attn_scale == pytest.approx(1.0 / 64)
        assert config.transformer_block_config.init_std == pytest.approx(0.02 / math.sqrt(2.0))
        assert config.output_scale == pytest.approx(1.0 / 2.0)

    def test_transformer_config_without_completep(self):
        """Test that TransformerConfig works normally without CompleteP."""
        config = TransformerConfig(
            name="test",
            hidden_dim=256,
            depth=4,
            transformer_block_config=TransformerBlockConfig(
                hidden_dim=256,
                num_heads=4,
                mlp_expansion_factor=4,
            ),
        )
        assert config.transformer_block_config.residual_scale == 1.0
        assert config.transformer_block_config.attn_scale is None
        assert config.transformer_block_config.init_std == 0.02
        assert config.output_scale == 1.0

    def test_alpha_half(self):
        """Test with depth_alpha_exp=0.5."""
        config = TransformerConfig(
            name="test",
            hidden_dim=512,
            depth=8,
            transformer_block_config=TransformerBlockConfig(
                hidden_dim=512,
                num_heads=8,
                mlp_expansion_factor=4,
            ),
            completep_config=CompletePConfig(
                base_width=256,
                base_depth=2,
                depth_alpha_exp=0.5,
            ),
        )
        m_d = 8 / 2  # = 4.0
        assert config.transformer_block_config.residual_scale == pytest.approx(1.0 / math.sqrt(4.0))


# =====================================================================================================================
# TransformerBlock residual scaling
# =====================================================================================================================


class TestResidualScale:
    def test_residual_scale_applied(self):
        """Test that residual_scale correctly scales single-block branch outputs."""
        torch.manual_seed(42)
        dim = 16
        # Use a single block so the residual relationship is exact:
        # out = x + scale * attn_branch + scale * mlp_branch(x + scale * attn_branch)
        # For a single block with Pre-LN: the branch only depends on LN(x) for attn,
        # but mlp depends on the attn-updated x. So we test with a simpler check.
        config_default = TransformerBlockConfig(hidden_dim=dim, num_heads=2, mlp_expansion_factor=4, residual_scale=1.0)
        block = TransformerBlock(config=config_default)

        x = torch.randn(2, 5, dim)

        # Test with scale=1.0
        block.residual_scale = 1.0
        out_default, _ = block(x)

        # Test with scale=0.0 (effectively skip all branches)
        block.residual_scale = 0.0
        out_zero, _ = block(x)
        # With scale=0, output should equal input
        # Note: residual_scale=0 isn't valid in config (gt=0.0) but works at runtime
        assert torch.allclose(out_zero, x, atol=1e-6)

        # Test that scale=1.0 differs from input (branches contribute)
        assert not torch.allclose(out_default, x, atol=1e-3)

    def test_residual_scale_gradient_flow(self):
        """Test gradient flow with residual_scale != 1."""
        dim = 16
        config = TransformerBlockConfig(hidden_dim=dim, num_heads=2, mlp_expansion_factor=4, residual_scale=0.25)
        block = TransformerBlock(config=config)
        x = torch.randn(2, 5, dim, requires_grad=True)
        out, _ = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# =====================================================================================================================
# Transformer output scaling
# =====================================================================================================================


class TestOutputScale:
    def test_output_scale_applied(self):
        """Test that output_scale is applied in Transformer forward."""
        torch.manual_seed(42)
        config_default = TransformerConfig(
            name="test",
            hidden_dim=16,
            depth=2,
            output_scale=1.0,
            transformer_block_config=TransformerBlockConfig(hidden_dim=16, num_heads=2, mlp_expansion_factor=4),
        )
        config_scaled = TransformerConfig(
            name="test",
            hidden_dim=16,
            depth=2,
            output_scale=0.5,
            transformer_block_config=TransformerBlockConfig(hidden_dim=16, num_heads=2, mlp_expansion_factor=4),
        )
        model_default = Transformer(config=config_default)
        model_scaled = Transformer(config=config_scaled)
        model_scaled.load_state_dict(model_default.state_dict())

        x = torch.randn(2, 5, 16)
        out_default = model_default(x, attn_kwargs={})
        out_scaled = model_scaled(x, attn_kwargs={})

        assert torch.allclose(out_scaled, out_default * 0.5, atol=1e-5)


# =====================================================================================================================
# CompletePModifier
# =====================================================================================================================


class SimpleTransformerModel(nn.Module):
    """Minimal model to test parameter classification."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 32)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(32),
                        "attn": nn.Linear(32, 32),
                        "norm2": nn.LayerNorm(32),
                        "mlp": nn.Linear(32, 32),
                    }
                )
                for _ in range(4)
            ]
        )
        self.final_norm = nn.LayerNorm(32)
        self.head = nn.Linear(32, 100, bias=False)


class TestCompletePModifier:
    @pytest.fixture
    def model(self):
        return SimpleTransformerModel()

    @pytest.fixture
    def modifier_alpha1(self):
        """CompleteP modifier with alpha=1, m_w=2, m_d=4."""
        cfg = CompletePModifierConfig(
            width_multiplier=2.0,
            depth_multiplier=4.0,
            depth_alpha_exp=1.0,
            base_eps=1e-8,
            base_weight_decay=0.1,
            hidden_param_substrings=["blocks."],
        )
        return CompletePModifier(cfg)

    def test_embedding_params(self, model, modifier_alpha1):
        """Embedding weight: lr_scale=1.0, wd=base_wd, eps=base_eps/m_w."""
        param = model.embedding.weight
        props = modifier_alpha1.get_properties(model, "embedding.weight", param)
        assert props["lr_scale"] == pytest.approx(1.0)
        assert props["weight_decay"] == pytest.approx(0.1)
        assert props["eps"] == pytest.approx(1e-8 / 2.0)

    def test_hidden_weight_params(self, model, modifier_alpha1):
        """Hidden weight (ndim>=2): lr_scale=1/m_w, wd=base_wd*m_w, eps=hidden_eps."""
        param = model.blocks[0]["attn"].weight
        props = modifier_alpha1.get_properties(model, "blocks.0.attn.weight", param)
        assert props["lr_scale"] == pytest.approx(1.0 / 2.0)  # width_lr * depth_lr = 0.5 * 1.0
        assert props["weight_decay"] == pytest.approx(0.1 * 2.0)
        assert props["eps"] == pytest.approx(1e-8 / 2.0 * (4.0**-1.0))

    def test_hidden_bias_params(self, model, modifier_alpha1):
        """Hidden bias: lr_scale=depth_lr (=1.0 for alpha=1), wd=0, eps=hidden_eps."""
        param = model.blocks[0]["attn"].bias
        props = modifier_alpha1.get_properties(model, "blocks.0.attn.bias", param)
        assert props["lr_scale"] == pytest.approx(1.0)  # m_d^(alpha-1) = 4^0 = 1
        assert props["weight_decay"] == pytest.approx(0.0)
        assert props["eps"] == pytest.approx(1e-8 / 2.0 * (4.0**-1.0))

    def test_hidden_norm_params(self, model, modifier_alpha1):
        """Hidden norm weight (ndim<=1, not bias): lr_scale=depth_lr, wd=0, eps=hidden_eps."""
        param = model.blocks[0]["norm1"].weight
        props = modifier_alpha1.get_properties(model, "blocks.0.norm1.weight", param)
        assert props["lr_scale"] == pytest.approx(1.0)
        assert props["weight_decay"] == pytest.approx(0.0)
        assert props["eps"] == pytest.approx(1e-8 / 2.0 * (4.0**-1.0))

    def test_final_norm_params(self, model, modifier_alpha1):
        """Final norm (outside blocks, ndim<=1): lr_scale=1.0, wd=0, eps=emb_eps."""
        param = model.final_norm.weight
        props = modifier_alpha1.get_properties(model, "final_norm.weight", param)
        assert props["lr_scale"] == pytest.approx(1.0)
        assert props["weight_decay"] == pytest.approx(0.0)
        assert props["eps"] == pytest.approx(1e-8 / 2.0)

    def test_head_params(self, model, modifier_alpha1):
        """Head weight (outside blocks, ndim>=2): lr_scale=1.0, wd=base_wd, eps=emb_eps."""
        param = model.head.weight
        props = modifier_alpha1.get_properties(model, "head.weight", param)
        assert props["lr_scale"] == pytest.approx(1.0)
        assert props["weight_decay"] == pytest.approx(0.1)
        assert props["eps"] == pytest.approx(1e-8 / 2.0)

    def test_alpha_half_depth_lr(self):
        """With alpha=0.5, depth_lr_scaling = m_d^(-0.5)."""
        cfg = CompletePModifierConfig(
            width_multiplier=1.0,
            depth_multiplier=4.0,
            depth_alpha_exp=0.5,
            base_eps=1e-8,
            base_weight_decay=0.1,
            hidden_param_substrings=["blocks."],
        )
        modifier = CompletePModifier(cfg)

        model = SimpleTransformerModel()
        param = model.blocks[0]["attn"].weight
        props = modifier.get_properties(model, "blocks.0.attn.weight", param)
        # depth_lr = m_d^(alpha-1) = 4^(-0.5) = 0.5
        # width_lr = 1/m_w = 1.0
        assert props["lr_scale"] == pytest.approx(0.5)

    def test_was_applied_successfully(self, model, modifier_alpha1):
        """Modifier should track that it was applied."""
        assert not modifier_alpha1.was_applied_successfully()
        modifier_alpha1.get_properties(model, "embedding.weight", model.embedding.weight)
        assert modifier_alpha1.was_applied_successfully()


# =====================================================================================================================
# CompletePModifier - Muon mode
# =====================================================================================================================


class TestCompletePModifierMuon:
    """Muon-targeted CompleteP: no ``eps`` anywhere, and hidden weight matrices receive only the
    depth LR scaling (no ``1/m_w`` width factor) and unscaled ``base_weight_decay``."""

    @pytest.fixture
    def model(self):
        return SimpleTransformerModel()

    @pytest.fixture
    def modifier_muon(self):
        cfg = CompletePModifierConfig(
            optimizer_type="muon",
            width_multiplier=2.0,
            depth_multiplier=4.0,
            depth_alpha_exp=1.0,
            base_eps=1e-8,
            base_weight_decay=0.1,
            hidden_param_substrings=["blocks."],
        )
        return CompletePModifier(cfg)

    def test_no_eps_returned_anywhere(self, model, modifier_muon):
        """In muon mode, ``eps`` must not appear in any param group's properties."""
        cases = [
            ("embedding.weight", model.embedding.weight),
            ("blocks.0.attn.weight", model.blocks[0]["attn"].weight),
            ("blocks.0.attn.bias", model.blocks[0]["attn"].bias),
            ("blocks.0.norm1.weight", model.blocks[0]["norm1"].weight),
            ("final_norm.weight", model.final_norm.weight),
            ("head.weight", model.head.weight),
        ]
        for name, param in cases:
            props = modifier_muon.get_properties(model, name, param)
            assert "eps" not in props, f"unexpected eps in muon-mode props for {name}: {props}"

    def test_hidden_weight_drops_width_scaling(self, model, modifier_muon):
        """Muon hidden weights: lr_scale = depth_lr only (no 1/m_w), wd = base_wd (no m_w)."""
        param = model.blocks[0]["attn"].weight
        props = modifier_muon.get_properties(model, "blocks.0.attn.weight", param)
        # depth_lr = m_d^(alpha-1) = 4^0 = 1
        assert props["lr_scale"] == pytest.approx(1.0)
        assert props["weight_decay"] == pytest.approx(0.1)

    def test_hidden_weight_keeps_depth_scaling(self):
        """With alpha=0.5 the depth factor still applies to hidden weights in muon mode."""
        cfg = CompletePModifierConfig(
            optimizer_type="muon",
            width_multiplier=2.0,
            depth_multiplier=4.0,
            depth_alpha_exp=0.5,
            base_eps=1e-8,
            base_weight_decay=0.1,
            hidden_param_substrings=["blocks."],
        )
        modifier = CompletePModifier(cfg)
        model = SimpleTransformerModel()
        props = modifier.get_properties(model, "blocks.0.attn.weight", model.blocks[0]["attn"].weight)
        # depth_lr = 4^(-0.5) = 0.5, no width factor in muon mode
        assert props["lr_scale"] == pytest.approx(0.5)
        assert props["weight_decay"] == pytest.approx(0.1)

    def test_embedding_unchanged_lr_and_wd(self, model, modifier_muon):
        """Embeddings have lr_scale=1.0 and base_wd in both modes."""
        props = modifier_muon.get_properties(model, "embedding.weight", model.embedding.weight)
        assert props["lr_scale"] == pytest.approx(1.0)
        assert props["weight_decay"] == pytest.approx(0.1)

    def test_hidden_norm_and_bias_keep_depth_scaling(self, model, modifier_muon):
        """1D hidden params still get depth_lr scaling and wd=0 in muon mode."""
        bias_props = modifier_muon.get_properties(model, "blocks.0.attn.bias", model.blocks[0]["attn"].bias)
        assert bias_props["lr_scale"] == pytest.approx(1.0)
        assert bias_props["weight_decay"] == pytest.approx(0.0)

        norm_props = modifier_muon.get_properties(model, "blocks.0.norm1.weight", model.blocks[0]["norm1"].weight)
        assert norm_props["lr_scale"] == pytest.approx(1.0)
        assert norm_props["weight_decay"] == pytest.approx(0.0)


# =====================================================================================================================
# CompletePModifier + MuonComposite end-to-end
# =====================================================================================================================


class TestCompletePMuonIntegration:
    """End-to-end: build MuonComposite from CompleteP-modified param groups and step it.

    Catches regressions where a stray ``eps`` (or other Adam-only kwarg) leaks into a Muon
    or Lion param group and misconfigures the optimizer.
    """

    def test_muon_composite_step_with_completep_groups(self):
        from noether.core.optimizer.muon_composite import MuonComposite

        model = SimpleTransformerModel()
        cfg = CompletePModifierConfig(
            optimizer_type="muon",
            width_multiplier=2.0,
            depth_multiplier=4.0,
            depth_alpha_exp=1.0,
            base_eps=1e-8,
            base_weight_decay=0.1,
            hidden_param_substrings=["blocks."],
        )
        modifier = CompletePModifier(cfg)

        param_groups = []
        for name, param in model.named_parameters():
            props = modifier.get_properties(model, name, param)
            assert "eps" not in props
            param_groups.append({"params": [param], "name": name, **props})

        opt = MuonComposite(param_groups, lr=1e-2, momentum=0.95, weight_decay=0.1)

        # No Muon or secondary group should have an ``eps`` key (CompleteP's eps must be filtered out).
        for g in opt._muon.param_groups:
            assert "eps" not in g or g["eps"] == opt._muon.defaults.get("eps")
        for g in opt._secondary.param_groups:
            assert "eps" not in g

        # A step with grads should run without error and update params.
        x = torch.randint(0, 100, (2, 5))
        emb = model.embedding(x)
        h = emb
        for blk in model.blocks:
            h = blk["mlp"](blk["attn"](blk["norm1"](h)))
        loss = model.head(model.final_norm(h)).sum()
        loss.backward()
        params_before = {n: p.clone() for n, p in model.named_parameters()}
        opt.step()
        assert any(not torch.equal(p, params_before[n]) for n, p in model.named_parameters())


# =====================================================================================================================
# Init std scaling
# =====================================================================================================================


class TestInitStdScaling:
    def test_init_std_propagated_to_mlp(self):
        """Test that init_std flows through to MLP weights."""
        torch.manual_seed(42)
        config = TransformerBlockConfig(
            hidden_dim=64,
            num_heads=4,
            mlp_expansion_factor=4,
            init_std=0.01,
        )
        block = TransformerBlock(config=config)
        # Check that MLP weights have std close to init_std
        fc1_std = block.mlp.fc1.weight.std().item()
        assert fc1_std < 0.02, f"Expected std < 0.02, got {fc1_std}"

    def test_default_init_std(self):
        """Test default init_std=0.02."""
        torch.manual_seed(42)
        config = TransformerBlockConfig(
            hidden_dim=256,
            num_heads=4,
            mlp_expansion_factor=4,
        )
        block = TransformerBlock(config=config)
        fc1_std = block.mlp.fc1.weight.std().item()
        assert abs(fc1_std - 0.02) < 0.005, f"Expected std near 0.02, got {fc1_std}"
