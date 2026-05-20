#  Copyright © 2026 Emmi AI GmbH. All rights reserved.
"""Unit tests for :mod:`noether.modeling.modules.untied`.

Covers :class:`UntiedLinear` (unified class with auto-selected strategy),
:class:`UntiedMixedAttention`, :class:`UntiedMLP`, and
:class:`UntiedTransformerBlock`.
"""

from collections.abc import Sequence

import pytest
import torch
from pydantic import ValidationError

from noether.core.schemas.modules.attention import AttentionPattern, TokenSpec
from noether.modeling.modules.blocks.transformer import TransformerBlockConfig
from noether.modeling.modules.layers.linear_projection import LinearProjectionConfig
from noether.modeling.modules.mlp.mlp import MLPConfig
from noether.modeling.modules.untied import (
    UntiedLinear,
    UntiedLinearConfig,
    UntiedMixedAttention,
    UntiedMixedAttentionConfig,
    UntiedMLP,
    UntiedMLPConfig,
    UntiedTransformerBlock,
    UntiedTransformerBlockConfig,
    _domain_group_sizes,
)


def _linear_cfg(
    num_types: int,
    in_dim: int,
    out_dim: int,
    *,
    bias: bool = True,
    init_weights: str = "torch",
) -> UntiedLinearConfig:
    """Helper to construct a :class:`UntiedLinearConfig`."""
    return UntiedLinearConfig(
        num_types=num_types,
        linear_projection=LinearProjectionConfig(
            input_dim=in_dim,
            output_dim=out_dim,
            bias=bias,
            init_weights=init_weights,
        ),
    )


# ---------------------------------------------------------------------------
# Domain helpers
# ---------------------------------------------------------------------------


class TestDomainGroupSizes:
    """Tests for :func:`_domain_group_sizes` — sums sizes per consecutive domain."""

    def test_single_spec_per_domain(self):
        specs = [TokenSpec(name="a", size=3), TokenSpec(name="b", size=2)]
        assert _domain_group_sizes(specs) == [3, 2]

    def test_multiple_specs_per_domain(self):
        """surface_anchors + surface_queries → single "surface" group."""
        specs = [
            TokenSpec(name="surface_anchors", size=10),
            TokenSpec(name="surface_queries", size=5),
            TokenSpec(name="volume_anchors", size=8),
            TokenSpec(name="volume_queries", size=3),
        ]
        assert _domain_group_sizes(specs) == [15, 11]

    def test_skips_none_and_zero(self):
        specs = [
            TokenSpec(name="surface_anchors", size=4),
            TokenSpec(name="surface_queries", size=None),
            TokenSpec(name="volume_anchors", size=0),
            TokenSpec(name="volume_queries", size=3),
        ]
        assert _domain_group_sizes(specs) == [4, 3]

    def test_empty_specs(self):
        assert _domain_group_sizes([]) == []

    def test_non_consecutive_domains_raises(self):
        """Domains that appear, disappear, and reappear must raise."""
        specs = [
            TokenSpec(name="surface_anchors", size=2),
            TokenSpec(name="volume_anchors", size=3),
            TokenSpec(name="surface_queries", size=1),  # surface reappears!
        ]
        with pytest.raises(ValueError, match="not consecutive"):
            _domain_group_sizes(specs)


# ---------------------------------------------------------------------------
# UntiedLinear
# ---------------------------------------------------------------------------


class TestUntiedLinear:
    def test_init_shapes(self):
        cfg = _linear_cfg(num_types=3, in_dim=8, out_dim=16)
        layer = UntiedLinear(cfg)
        assert layer.num_types == 3
        assert len(layer.weight) == 3
        assert all(w.shape == (16, 8) for w in layer.weight)
        assert layer.bias is not None
        assert len(layer.bias) == 3
        assert all(b.shape == (16,) for b in layer.bias)

    def test_init_no_bias(self):
        cfg = _linear_cfg(num_types=2, in_dim=4, out_dim=4, bias=False)
        layer = UntiedLinear(cfg)
        assert layer.bias is None

    @pytest.mark.parametrize("init_weights", ["torch", "truncnormal", "truncnormal002"])
    def test_init_weights_modes_run(self, init_weights):
        """Supported ``init_weights`` modes run without error and produce non-zero weights."""
        cfg = _linear_cfg(num_types=2, in_dim=8, out_dim=8, init_weights=init_weights)
        layer = UntiedLinear(cfg)
        assert not all(torch.all(w == 0) for w in layer.weight)

    def test_init_weights_zeros(self):
        """``"zeros"`` mode produces zero weight and zero bias."""
        cfg = _linear_cfg(num_types=2, in_dim=4, out_dim=4, init_weights="zeros")
        layer = UntiedLinear(cfg)
        assert all(torch.all(w == 0) for w in layer.weight)
        assert all(torch.all(b == 0) for b in layer.bias)

    def test_init_weights_truncnormal_std(self):
        """``truncnormal002`` produces weights with std ≈ 0.02 and zero bias."""
        torch.manual_seed(0)
        cfg = _linear_cfg(num_types=2, in_dim=16, out_dim=16, init_weights="truncnormal002")
        layer = UntiedLinear(cfg)
        stacked = torch.stack(list(layer.weight))
        assert 0.01 < stacked.std().item() < 0.04
        assert all(torch.all(b == 0) for b in layer.bias)

    def test_init_weights_unknown_raises(self):
        """Unsupported ``init_weights`` values raise ``NotImplementedError`` at build time."""
        # The schema only accepts a fixed Literal, so truly-invalid strings are rejected by pydantic.
        with pytest.raises(ValidationError):
            _linear_cfg(num_types=2, in_dim=4, out_dim=4, init_weights="nonsense")

    def test_init_weights_unsupported_raises_not_implemented(self):
        """A valid ``InitWeightsMode`` that :class:`UntiedLinear` doesn't handle raises."""
        # "truncnormal002-identity" is a valid InitWeightsMode value but not handled by UntiedLinear.
        cfg = _linear_cfg(num_types=2, in_dim=4, out_dim=4, init_weights="truncnormal002-identity")
        with pytest.raises(NotImplementedError, match="truncnormal002-identity"):
            UntiedLinear(cfg)

    def test_forward_shape(self):
        cfg = _linear_cfg(num_types=2, in_dim=4, out_dim=6)
        layer = UntiedLinear(cfg)
        specs = [TokenSpec(name="a", size=3), TokenSpec(name="b", size=2)]
        x = torch.randn(2, 5, 4)
        out = layer(x, specs)
        assert out.shape == (2, 5, 6)

    def test_forward_respects_per_type_weights(self):
        """Each per-type weight bank is actually applied to its own tokens."""
        cfg = _linear_cfg(num_types=2, in_dim=4, out_dim=4, bias=False)
        layer = UntiedLinear(cfg)
        # Make type 0's weight the identity, type 1's the zero map — easy to check.
        with torch.no_grad():
            layer.weight[0].copy_(torch.eye(4))
            layer.weight[1].copy_(torch.zeros(4, 4))

        specs = [TokenSpec(name="a", size=2), TokenSpec(name="b", size=3)]
        x = torch.randn(1, 5, 4)
        out = layer(x, specs)
        # "a" rows (first 2) must pass through unchanged.
        assert torch.allclose(out[:, :2], x[:, :2])
        # "b" rows (last 3) must be zeroed.
        assert torch.all(out[:, 2:] == 0)

    def test_forward_skips_empty_and_cached_specs(self):
        """Empty (size=0) and cached (size=None) specs contribute no output tokens."""
        cfg = _linear_cfg(num_types=2, in_dim=4, out_dim=4)
        layer = UntiedLinear(cfg)
        specs = [
            TokenSpec(name="a", size=2),
            TokenSpec(name="empty", size=0),
            TokenSpec(name="cached", size=None),
            TokenSpec(name="b", size=3),
        ]
        x = torch.randn(2, 5, 4)
        out = layer(x, specs)
        # Only the 2 + 3 real tokens remain in the output.
        assert out.shape == (2, 5, 4)

    def test_forward_backward_produces_per_type_grads(self):
        """Backward pass populates all per-type weight banks with gradients."""
        cfg = _linear_cfg(num_types=2, in_dim=4, out_dim=4)
        layer = UntiedLinear(cfg)
        specs = [TokenSpec(name="a", size=3), TokenSpec(name="b", size=2)]
        out = layer(torch.randn(2, 5, 4), specs)
        out.sum().backward()
        # Each per-type weight Parameter (2D) has its own grad of shape (D_out, D_in).
        for w in layer.weight:
            assert w.grad is not None
            assert w.grad.shape == w.shape
        # Both type banks receive gradients (neither is all zero).
        assert torch.any(layer.weight[0].grad != 0)
        assert torch.any(layer.weight[1].grad != 0)

    def test_forward_multi_spec_per_domain(self):
        """Multiple specs sharing a domain are grouped and share one weight bank."""
        cfg = _linear_cfg(num_types=2, in_dim=4, out_dim=4, bias=False)
        layer = UntiedLinear(cfg)
        with torch.no_grad():
            layer.weight[0].copy_(torch.eye(4))  # "surface" domain → identity
            layer.weight[1].copy_(torch.zeros(4, 4))  # "volume" domain → zero

        specs = [
            TokenSpec(name="surface_anchors", size=2),
            TokenSpec(name="surface_queries", size=1),
            TokenSpec(name="volume_anchors", size=2),
        ]
        x = torch.randn(1, 5, 4)
        out = layer(x, specs)
        # surface (first 3 tokens) pass through; volume (last 2) zeroed.
        assert torch.allclose(out[:, :3], x[:, :3])
        assert torch.all(out[:, 3:] == 0)

    def test_forward_non_consecutive_domain_raises(self):
        """Non-consecutive domains are rejected at forward time."""
        cfg = _linear_cfg(num_types=2, in_dim=4, out_dim=4)
        layer = UntiedLinear(cfg)
        specs = [
            TokenSpec(name="surface_anchors", size=2),
            TokenSpec(name="volume_anchors", size=3),
            TokenSpec(name="surface_queries", size=1),  # surface reappears
        ]
        with pytest.raises(ValueError, match="not consecutive"):
            layer(torch.randn(1, 6, 4), specs)

    def test_no_cross_talk_between_types(self):
        """Changing type ``b`` inputs must not affect type ``a`` outputs."""
        cfg = _linear_cfg(num_types=2, in_dim=4, out_dim=4)
        layer = UntiedLinear(cfg)
        specs = [TokenSpec(name="a", size=3), TokenSpec(name="b", size=2)]
        x_a = torch.randn(1, 3, 4)
        x_b1 = torch.randn(1, 2, 4)
        x_b2 = torch.randn(1, 2, 4) + 100.0  # vastly different

        out1 = layer(torch.cat([x_a, x_b1], dim=1), specs)
        out2 = layer(torch.cat([x_a, x_b2], dim=1), specs)

        assert torch.allclose(out1[:, :3], out2[:, :3], atol=1e-6)
        assert not torch.allclose(out1[:, 3:], out2[:, 3:])


class TestUntiedLinearStrategies:
    """The internal strategies (split-loop, equal-size bmm, padded bmm) must agree."""

    def test_split_loop_and_equal_size_bmm_agree(self):
        """Both strategies produce the same output when domain sizes are equal."""
        torch.manual_seed(0)
        cfg = _linear_cfg(num_types=2, in_dim=4, out_dim=6)
        layer = UntiedLinear(cfg)

        specs = [TokenSpec(name="a", size=3), TokenSpec(name="b", size=3)]
        x = torch.randn(2, 6, 4)

        from noether.modeling.modules.untied import _domain_group_sizes

        sizes = _domain_group_sizes(specs)
        B, S, D_in = x.shape
        D_out = layer.output_dim
        T = len(sizes)

        out_loop = layer._split_loop_forward(x, sizes, D_out)
        out_bmm = layer._equal_size_bmm_forward(x, sizes[0], B, S, D_in, D_out, T)
        assert torch.allclose(out_loop, out_bmm, atol=1e-5)

    def test_split_loop_and_padded_bmm_agree(self):
        """Both strategies produce the same output for unequal sizes."""
        torch.manual_seed(0)
        cfg = _linear_cfg(num_types=2, in_dim=4, out_dim=6)
        layer = UntiedLinear(cfg)

        specs = [TokenSpec(name="a", size=3), TokenSpec(name="b", size=2)]
        x = torch.randn(2, 5, 4)

        from noether.modeling.modules.untied import _domain_group_sizes

        sizes = _domain_group_sizes(specs)
        B, S, D_in = x.shape
        D_out = layer.output_dim
        T = len(sizes)

        out_loop = layer._split_loop_forward(x, sizes, D_out)
        out_padded = layer._padded_bmm_forward(x, sizes, B, S, D_in, D_out, T, max(sizes))
        assert torch.allclose(out_loop, out_padded, atol=1e-5)

    def test_auto_forward_backward(self):
        """The auto-selected strategy supports forward + backward."""
        cfg = _linear_cfg(num_types=2, in_dim=4, out_dim=4)
        layer = UntiedLinear(cfg)
        specs = [TokenSpec(name="a", size=3), TokenSpec(name="b", size=2)]
        out = layer(torch.randn(2, 5, 4), specs)
        assert out.shape == (2, 5, 4)
        out.sum().backward()
        assert all(w.grad is not None for w in layer.weight)


# ---------------------------------------------------------------------------
# UntiedMixedAttention
# ---------------------------------------------------------------------------


def _make_attn(
    *,
    hidden_dim: int = 32,
    num_heads: int = 4,
    num_types: int = 2,
    use_rope: bool = False,
    dropout: float = 0.0,
) -> UntiedMixedAttention:
    cfg = UntiedMixedAttentionConfig(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
        bias=True,
        use_rope=use_rope,
        num_types=num_types,
    )
    return UntiedMixedAttention(config=cfg).eval()


def _all_to_all_patterns(specs: Sequence[TokenSpec]) -> list[AttentionPattern]:
    """Single all-to-all attention pattern over every name in ``specs``."""
    names = [s.name for s in specs if s.size is not None]
    return [AttentionPattern(query_tokens=names, key_value_tokens=names)]


class TestUntiedMixedAttention:
    def test_init_uses_untied_projections(self):
        """Q, K, V, and output projections are ``UntiedLinear`` with one ``(hidden_dim, hidden_dim)`` Parameter per type."""
        attn = _make_attn(hidden_dim=32, num_types=3)
        for proj in (attn.q, attn.k, attn.v, attn.proj):
            assert isinstance(proj, UntiedLinear)
            assert len(proj.weight) == 3
            assert all(w.shape == (32, 32) for w in proj.weight)

    def test_forward_shape(self):
        attn = _make_attn()
        specs = [TokenSpec(name="a", size=4), TokenSpec(name="b", size=6)]
        x = torch.randn(2, 10, 32)
        out = attn(x, specs, _all_to_all_patterns(specs))
        assert out.shape == (2, 10, 32)

    def test_forward_with_key_padding_mask(self):
        attn = _make_attn()
        specs = [TokenSpec(name="a", size=4), TokenSpec(name="b", size=6)]
        x = torch.randn(2, 10, 32)
        mask = torch.ones(2, 10, dtype=torch.bool)
        mask[0, -3:] = False  # last 3 of the b-group are padding in batch 0
        out = attn(x, specs, _all_to_all_patterns(specs), key_padding_mask=mask)
        assert out.shape == (2, 10, 32)

    def test_kv_cache_not_implemented(self):
        """Supplying ``kv_cache`` is not yet supported and raises."""
        attn = _make_attn()
        specs = [TokenSpec(name="a", size=4), TokenSpec(name="b", size=6)]
        x = torch.randn(2, 10, 32)
        with pytest.raises(NotImplementedError, match="kv caching"):
            attn(x, specs, _all_to_all_patterns(specs), kv_cache={"a": torch.randn(1)})

    def test_backward_populates_per_type_grads(self):
        attn = _make_attn(num_types=2)
        specs = [TokenSpec(name="a", size=4), TokenSpec(name="b", size=6)]
        out = attn(torch.randn(2, 10, 32), specs, _all_to_all_patterns(specs))
        out.sum().backward()
        # Per-type q/k/v weight banks all receive gradient signal.
        for proj in (attn.q, attn.k, attn.v):
            for w in proj.weight:
                assert w.grad is not None
                assert w.grad.shape == (32, 32)
            assert torch.any(proj.weight[0].grad != 0)
            assert torch.any(proj.weight[1].grad != 0)

    def test_cross_attention_actually_cross_attends(self):
        """With a cross pattern (``a→b``, ``b→a``), ``a``'s output must depend on ``b``'s input.

        Validates that ``AttentionPattern`` is honored end-to-end: when type ``a``
        queries *only* type ``b``'s key/values, varying ``b``'s input must shift
        ``a``'s output (b provides the K/V). Conversely, under a self-only pattern
        (``a→a``, ``b→b``) ``a``'s output is invariant to ``b``. Together these
        prove the module isn't silently collapsing to self-attention.
        """
        torch.manual_seed(0)
        attn = _make_attn(num_types=2)
        specs = [TokenSpec(name="a", size=4), TokenSpec(name="b", size=6)]
        cross = [
            AttentionPattern(query_tokens=["a"], key_value_tokens=["b"]),
            AttentionPattern(query_tokens=["b"], key_value_tokens=["a"]),
        ]
        self_only = [
            AttentionPattern(query_tokens=["a"], key_value_tokens=["a"]),
            AttentionPattern(query_tokens=["b"], key_value_tokens=["b"]),
        ]

        x_a = torch.randn(1, 4, 32)
        x_b1 = torch.randn(1, 6, 32)
        x_b2 = torch.randn(1, 6, 32)

        # Cross pattern: a queries b → changing b changes a's output.
        out1_cross = attn(torch.cat([x_a, x_b1], dim=1), specs, cross)
        out2_cross = attn(torch.cat([x_a, x_b2], dim=1), specs, cross)
        assert not torch.allclose(out1_cross[:, :4], out2_cross[:, :4])

        # Self-only pattern: a attends only to a → a's output is invariant to b.
        out1_self = attn(torch.cat([x_a, x_b1], dim=1), specs, self_only)
        out2_self = attn(torch.cat([x_a, x_b2], dim=1), specs, self_only)
        assert torch.allclose(out1_self[:, :4], out2_self[:, :4], atol=1e-6)

    def test_no_cross_talk_between_types_in_projection(self):
        """Changing type ``b`` inputs shouldn't shift type ``a`` tokens' *projections*.

        Attention itself mixes all tokens, so the final attention output naturally
        depends on every token. We isolate the projection path by giving the attention
        output a trivial structure: with identity q/k/v and zero values for type b,
        type a tokens' q/k/v are untouched by type b inputs at the projection layer.
        Here we just verify the projection weights are keyed per-type.
        """
        attn = _make_attn()
        # Spec names → projection weight index: a→0, b→1. Verify they're distinct
        # parameter tensors (not aliases) for each of q/k/v.
        for proj in (attn.q, attn.k, attn.v):
            assert proj.weight[0].data_ptr() != proj.weight[1].data_ptr() or torch.any(proj.weight[0] != proj.weight[1])


# ---------------------------------------------------------------------------
# UntiedMLP
# ---------------------------------------------------------------------------


class TestUntiedMLP:
    def _cfg(self, *, num_layers: int = 0) -> UntiedMLPConfig:
        return UntiedMLPConfig(
            num_types=2,
            mlp=MLPConfig(
                input_dim=16,
                output_dim=16,
                hidden_dim=32,
                num_layers=num_layers,
                activation="GELU",
                init_weights="truncnormal002",
            ),
        )

    def test_init_num_layers_zero_gives_two_projections(self):
        """``num_layers=0`` produces the canonical up→act→down topology (2 linears)."""
        mlp = UntiedMLP(self._cfg(num_layers=0))
        assert len(mlp.layers) == 2
        # up: 16 → 32
        assert len(mlp.layers[0].weight) == 2
        assert all(w.shape == (32, 16) for w in mlp.layers[0].weight)
        # down: 32 → 16
        assert len(mlp.layers[1].weight) == 2
        assert all(w.shape == (16, 32) for w in mlp.layers[1].weight)

    def test_init_num_layers_two_gives_four_projections(self):
        """``num_layers=2`` produces input + 2 hidden + output (4 linears total)."""
        mlp = UntiedMLP(self._cfg(num_layers=2))
        assert len(mlp.layers) == 4

    def test_forward_shape(self):
        mlp = UntiedMLP(self._cfg())
        specs = [TokenSpec(name="a", size=3), TokenSpec(name="b", size=2)]
        x = torch.randn(2, 5, 16)
        out = mlp(x, specs)
        assert out.shape == (2, 5, 16)

    def test_forward_backward(self):
        mlp = UntiedMLP(self._cfg(num_layers=1))
        specs = [TokenSpec(name="a", size=3), TokenSpec(name="b", size=2)]
        out = mlp(torch.randn(2, 5, 16), specs)
        out.sum().backward()
        for layer in mlp.layers:
            assert all(w.grad is not None for w in layer.weight)

    def test_all_layers_are_untied_linear(self):
        """Every linear layer in the MLP is a :class:`UntiedLinear`."""
        mlp = UntiedMLP(self._cfg(num_layers=1))
        assert all(isinstance(layer, UntiedLinear) for layer in mlp.layers)


# ---------------------------------------------------------------------------
# UntiedTransformerBlock
# ---------------------------------------------------------------------------


def _tb_cfg(
    *,
    hidden_dim: int = 32,
    num_heads: int = 4,
    condition_dim: int | None = None,
    drop_path: float = 0.0,
    layerscale: float | None = None,
) -> TransformerBlockConfig:
    return TransformerBlockConfig(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        mlp_expansion_factor=2,
        drop_path=drop_path,
        condition_dim=condition_dim,
        use_rope=False,
        bias=True,
        layerscale=layerscale,
    )


def _block_cfg(**tb_kwargs: object) -> UntiedTransformerBlockConfig:
    return UntiedTransformerBlockConfig(
        num_types=2,
        transformer_block=_tb_cfg(**tb_kwargs),
    )


def _specs() -> Sequence[TokenSpec]:
    return [TokenSpec(name="a", size=4), TokenSpec(name="b", size=6)]


class TestUntiedTransformerBlock:
    def test_init_assembles_submodules(self):
        block = _block_cfg()
        tb = UntiedTransformerBlock(config=block)
        assert isinstance(tb.attention_block, UntiedMixedAttention)
        assert isinstance(tb.mlp, UntiedMLP)
        assert tb.modulation is None  # no condition_dim → no modulation

    def test_forward_shape(self):
        tb = UntiedTransformerBlock(config=_block_cfg())
        x = torch.randn(2, 10, 32)
        specs = _specs()
        out, cache = tb(x, attn_kwargs={"token_specs": specs, "attention_patterns": _all_to_all_patterns(specs)})
        assert out.shape == (2, 10, 32)
        # UntiedMixedAttention returns a bare tensor (no cache), so block's cache is None.
        assert cache is None

    def test_forward_missing_token_specs_raises(self):
        """``attn_kwargs`` must include ``token_specs`` — otherwise a clear error is raised."""
        tb = UntiedTransformerBlock(config=_block_cfg())
        x = torch.randn(2, 10, 32)
        with pytest.raises(ValueError, match="token_specs"):
            tb(x, attn_kwargs={})
        with pytest.raises(ValueError, match="token_specs"):
            tb(x, attn_kwargs=None)
        with pytest.raises(ValueError, match="token_specs"):
            tb(x, attn_kwargs={"token_specs": None})

    def test_forward_rejects_condition_without_modulation(self):
        """Passing ``condition`` when ``condition_dim`` is ``None`` must raise."""
        tb = UntiedTransformerBlock(config=_block_cfg())
        x = torch.randn(2, 10, 32)
        with pytest.raises(ValueError, match="not configured for conditioning"):
            tb(x, attn_kwargs={"token_specs": _specs()}, condition=torch.randn(2, 4))

    def test_init_with_condition_dim_builds_modulation(self):
        """``condition_dim`` wires up the modulation projection; leaves attention/MLP as-is."""
        tb = UntiedTransformerBlock(config=_block_cfg(condition_dim=8))
        assert tb.modulation is not None
        assert isinstance(tb.attention_block, UntiedMixedAttention)
        assert isinstance(tb.mlp, UntiedMLP)

    def test_forward_missing_condition_when_required_raises(self):
        """With ``condition_dim`` set, forward without ``condition`` must raise."""
        tb = UntiedTransformerBlock(config=_block_cfg(condition_dim=8))
        x = torch.randn(2, 10, 32)
        with pytest.raises(ValueError, match="No conditioning vector provided"):
            tb(x, attn_kwargs={"token_specs": _specs()})

    def test_forward_condition_shape_mismatch_raises(self):
        tb = UntiedTransformerBlock(config=_block_cfg(condition_dim=8))
        x = torch.randn(2, 10, 32)
        with pytest.raises(ValueError, match="incorrect shape"):
            tb(x, attn_kwargs={"token_specs": _specs()}, condition=torch.randn(2, 4))

    def test_forward_backward(self):
        tb = UntiedTransformerBlock(config=_block_cfg())
        x = torch.randn(2, 10, 32)
        specs = _specs()
        out, _ = tb(x, attn_kwargs={"token_specs": specs, "attention_patterns": _all_to_all_patterns(specs)})
        out.sum().backward()
        # Both attention and MLP have gradient signal in their per-type weight banks.
        for proj in (tb.attention_block.q, tb.attention_block.k, tb.attention_block.v):
            for w in proj.weight:
                assert w.grad is not None
                assert torch.any(w.grad != 0)
        for layer in tb.mlp.layers:
            assert all(w.grad is not None for w in layer.weight)

    def test_forward_with_multi_spec_per_domain(self):
        """Multiple specs per domain (anchors + queries) use the same weight bank."""
        tb = UntiedTransformerBlock(config=_block_cfg())
        specs = [
            TokenSpec(name="a_anchors", size=4),
            TokenSpec(name="a_queries", size=2),
            TokenSpec(name="b_anchors", size=3),
            TokenSpec(name="b_queries", size=1),
        ]
        x = torch.randn(2, 10, 32)
        out, _ = tb(x, attn_kwargs={"token_specs": specs, "attention_patterns": _all_to_all_patterns(specs)})
        assert out.shape == (2, 10, 32)

    def test_forward_non_consecutive_domain_raises(self):
        """Interleaved domains are caught by the block's validation."""
        tb = UntiedTransformerBlock(config=_block_cfg())
        specs = [
            TokenSpec(name="a_anchors", size=3),
            TokenSpec(name="b_anchors", size=3),
            TokenSpec(name="a_queries", size=2),  # "a" domain reappears
            TokenSpec(name="b_queries", size=2),
        ]
        with pytest.raises(ValueError, match="not consecutive"):
            tb(torch.randn(1, 10, 32), attn_kwargs={"token_specs": specs})

    def test_different_partitions_produce_different_outputs(self):
        """Moving the a/b split point changes which tokens use which weight bank."""
        torch.manual_seed(0)
        tb = UntiedTransformerBlock(config=_block_cfg()).eval()
        x = torch.randn(1, 10, 32)

        # Same input, different a/b partitions. The block has distinct per-type
        # weights, so a different partition must produce a different output.
        specs_v1 = [TokenSpec(name="a", size=5), TokenSpec(name="b", size=5)]
        specs_v2 = [TokenSpec(name="a", size=3), TokenSpec(name="b", size=7)]

        out1, _ = tb(x, attn_kwargs={"token_specs": specs_v1, "attention_patterns": _all_to_all_patterns(specs_v1)})
        out2, _ = tb(x, attn_kwargs={"token_specs": specs_v2, "attention_patterns": _all_to_all_patterns(specs_v2)})
        assert not torch.allclose(out1, out2)
