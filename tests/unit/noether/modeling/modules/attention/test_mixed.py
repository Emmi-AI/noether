#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError

from noether.core.schemas.modules.attention import AttentionPattern, MixedAttentionConfig, TokenSpec
from noether.modeling.modules.attention.anchor_attention.mixed import MixedAttention


@pytest.fixture
def mock_config():
    return MixedAttentionConfig(
        hidden_dim=64,
        num_heads=4,
        dropout=0.1,
        bias=True,
        use_rope=False,
    )


class TestTokenSpec:
    def test_valid_creation(self):
        spec = TokenSpec(name="surface_anchors", size=100)
        assert spec.name == "surface_anchors"
        assert spec.size == 100

    def test_invalid_name(self):
        with pytest.raises(ValidationError):
            TokenSpec(name="invalid_name", size=100)  # type: ignore

    def test_negative_size(self):
        with pytest.raises(ValidationError):
            TokenSpec(name="surface_anchors", size=-1)

    def test_from_dict(self):
        spec = TokenSpec.from_dict({"volume_queries": 50})
        assert spec.name == "volume_queries"
        assert spec.size == 50

    def test_to_dict(self):
        spec = TokenSpec(name="surface_queries", size=10)
        assert spec.to_dict() == {"surface_queries": 10}


class TestMixedAttention:
    @pytest.fixture
    def module(self, mock_config):
        return MixedAttention(config=mock_config).eval()

    def test_init(self, mock_config):
        module = MixedAttention(config=mock_config).eval()
        assert isinstance(module, MixedAttention)
        assert module.num_heads == 4
        assert module.head_dim == 16

    def test_forward_basic_pattern_isolation(self, module):
        """
        Verify that token groups are actually isolated.
        If surface_anchors only attend to themselves, changing the surface_queries
        should NOT affect the surface_anchors' output.
        """
        batch_size = 2
        dim = 64

        surface_anchors = torch.randn(batch_size, 10, dim)
        surface_queries_1 = torch.randn(batch_size, 5, dim)
        surface_queries_2 = torch.randn(batch_size, 5, dim) + 100.0  # vastly different values

        token_specs = [
            TokenSpec(name="surface_anchors", size=10),
            TokenSpec(name="surface_queries", size=5),
        ]

        patterns = [
            AttentionPattern(query_tokens=["surface_anchors"], key_value_tokens=["surface_anchors"]),
            AttentionPattern(query_tokens=["surface_queries"], key_value_tokens=["surface_queries"]),
        ]

        # Run twice with SAME anchors but DIFFERENT queries:
        input1 = torch.cat([surface_anchors, surface_queries_1], dim=1)
        output1 = module(input1, token_specs=token_specs, attention_patterns=patterns)

        input2 = torch.cat([surface_anchors, surface_queries_2], dim=1)
        output2 = module(input2, token_specs=token_specs, attention_patterns=patterns)

        assert torch.allclose(output1[:, :10], output2[:, :10], atol=1e-6)

        # The query part SHOULD be different:
        assert not torch.allclose(output1[:, 10:], output2[:, 10:])

    def test_forward_mixed_interaction(self, module):
        """
        Test complex interaction:
        - surface_queries (5) attend to surface_anchors (10)
        - surface_anchors (10) attend to surface_anchors (10)
        """
        x = torch.randn(1, 15, 64)

        token_specs = [
            TokenSpec(name="surface_queries", size=5),
            TokenSpec(name="surface_anchors", size=10),
        ]

        patterns = [
            AttentionPattern(query_tokens=["surface_queries"], key_value_tokens=["surface_anchors"]),
            AttentionPattern(query_tokens=["surface_anchors"], key_value_tokens=["surface_anchors"]),
        ]

        output = module(x, token_specs, patterns)
        assert output.shape == (1, 15, 64)

    def test_process_pattern_batched_optimization(self, module):
        """
        Verify that patterns with identical shapes are batched together.
        """
        dim = 64
        # We define 4 tokens, all size 5.
        # To satisfy validation, ALL 4 must be queries in some pattern.
        # We create 4 patterns, all with Q_len=5 and KV_len=5.
        # These should all be grouped into a single batch.

        x = torch.randn(1, 20, dim)

        token_specs = [
            TokenSpec(name="surface_anchors", size=5),
            TokenSpec(name="volume_anchors", size=5),
            TokenSpec(name="surface_queries", size=5),
            TokenSpec(name="volume_queries", size=5),
        ]

        patterns = [
            # 1. SA -> VA:
            AttentionPattern(query_tokens=["surface_anchors"], key_value_tokens=["volume_anchors"]),
            # 2. SQ -> VQ:
            AttentionPattern(query_tokens=["surface_queries"], key_value_tokens=["volume_queries"]),
            # 3. VA -> SA (added to make VA a query):
            AttentionPattern(query_tokens=["volume_anchors"], key_value_tokens=["surface_anchors"]),
            # 4. VQ -> SQ (added to make VQ a query):
            AttentionPattern(query_tokens=["volume_queries"], key_value_tokens=["surface_queries"]),
        ]

        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            # Expected batched shape:
            # Batch dimension = original_batch(1) * num_patterns(4) = 4
            # Q len = 5, Head dim = 16
            mock_sdpa.return_value = torch.randn(4, 4, 5, 16)

            module(x, token_specs, patterns)
            assert mock_sdpa.call_count == 1

            args, _ = mock_sdpa.call_args
            assert args[0].shape[0] == 4

    def test_validation_errors(self, module):
        dim = 64
        x = torch.randn(1, 10, dim)

        with pytest.raises(ValueError, match="Token specs total size"):
            module(
                x,
                token_specs=[TokenSpec(name="surface_anchors", size=5)],
                attention_patterns=[
                    AttentionPattern(query_tokens=["surface_anchors"], key_value_tokens=["surface_anchors"]),
                ],
            )

        token_specs = [
            TokenSpec(name="surface_anchors", size=5),
            TokenSpec(name="surface_queries", size=5),
        ]
        with pytest.raises(ValueError, match="set of query tokens must exactly match"):
            module(
                x,
                token_specs=token_specs,
                attention_patterns=[
                    AttentionPattern(query_tokens=["surface_anchors"], key_value_tokens=["surface_anchors"]),
                ],
            )

        with pytest.raises(ValueError, match="cannot be a query in multiple"):
            module(
                x,
                token_specs=token_specs,
                attention_patterns=[
                    AttentionPattern(query_tokens=["surface_anchors"], key_value_tokens=["surface_anchors"]),
                    AttentionPattern(
                        query_tokens=["surface_anchors", "surface_queries"], key_value_tokens=["surface_queries"]
                    ),
                ],
            )

        with pytest.raises(ValueError, match="is not defined in `token_specs`"):
            module(
                x,
                token_specs=[TokenSpec(name="surface_anchors", size=10)],
                attention_patterns=[
                    AttentionPattern(query_tokens=["surface_anchors"], key_value_tokens=["volume_anchors"]),
                ],
            )

    def test_concatenation_order(self, module):
        """
        Verify that multiple query tokens in one pattern are concatenated correctly.
        Pattern: [surface_anchors, surface_queries] attend to [volume_anchors].
        """
        # Sizes: 5 + 5 + 5 = 15
        x = torch.randn(1, 15, 64)
        token_specs = [
            TokenSpec(name="surface_anchors", size=5),
            TokenSpec(name="surface_queries", size=5),
            TokenSpec(name="volume_anchors", size=5),
        ]

        patterns = [
            # Q = surface_anchors + surface_queries (len 10):
            AttentionPattern(query_tokens=["surface_anchors", "surface_queries"], key_value_tokens=["volume_anchors"]),
            # Q = volume_anchors (len 5)
            # Must include volume_anchors as query to be valid:
            AttentionPattern(query_tokens=["volume_anchors"], key_value_tokens=["volume_anchors"]),
        ]

        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            # Mock return for Q=10 and Q=5. Lengths differ, so NO batching expected.
            def side_effect(q, k, v, **kwargs):
                return torch.randn(q.shape[0], q.shape[1], q.shape[2], q.shape[3])

            mock_sdpa.side_effect = side_effect

            module(x, token_specs, patterns)
            assert mock_sdpa.call_count == 2

            q_lens = [call[0][0].shape[2] for call in mock_sdpa.call_args_list]
            assert 10 in q_lens
            assert 5 in q_lens

    def test_rope_integration(self):
        config = MixedAttentionConfig(
            hidden_dim=64,
            num_heads=4,
            bias=True,
            use_rope=True,  # ENABLED
        )
        module = MixedAttention(config)

        x = torch.randn(1, 10, 64)
        token_specs = [TokenSpec(name="surface_anchors", size=10)]
        patterns = [AttentionPattern(query_tokens=["surface_anchors"], key_value_tokens=["surface_anchors"])]

        # 1. Should fail if rope is enabled but freqs is None:
        with pytest.raises(ValueError, match="RoPE usage mismatch"):
            module(x, token_specs, patterns, freqs=None)

        # 2. Should pass with freqs:
        freqs = torch.randn(10, 8)

        with patch("noether.modeling.modules.attention.anchor_attention.mixed.rope") as mock_rope:
            mock_rope.side_effect = lambda t, freqs: t  # Identity mock

    def test_attn_mask_validation_errors(self, module):
        x = torch.randn(1, 10, 64)
        token_specs = [TokenSpec(name="surface_anchors", size=10)]
        patterns = [AttentionPattern(query_tokens=["surface_anchors"], key_value_tokens=["surface_anchors"])]

        with pytest.raises(ValueError, match="bool tensor"):
            module(x, token_specs, patterns, key_padding_mask=torch.ones(1, 10))  # float, not bool

        with pytest.raises(ValueError, match="2D"):
            module(x, token_specs, patterns, key_padding_mask=torch.ones(1, 10, 1, dtype=torch.bool))

        with pytest.raises(ValueError, match="n_tokens dim"):
            module(x, token_specs, patterns, key_padding_mask=torch.ones(1, 5, dtype=torch.bool))

    def test_attn_mask_matches_unpadded_run(self, module):
        """Masked batched run must match individual unpadded runs for each batch element.

        Three items with different numbers of real anchors are padded to size 3 and run
        together with a mask. Outputs at real positions must equal running each item alone
        without any mask or padding.
        """
        torch.manual_seed(42)
        dim = 64

        anchors_0 = torch.randn(1, 3, dim)  # 3 real anchors (no padding)
        queries_0 = torch.randn(1, 4, dim)
        anchors_1 = torch.randn(1, 2, dim)  # 2 real anchors, 1 padding
        queries_1 = torch.randn(1, 4, dim)
        anchors_2 = torch.randn(1, 1, dim)  # 1 real anchor, 2 padding
        queries_2 = torch.randn(1, 4, dim)

        token_specs_padded = [
            TokenSpec(name="surface_anchors", size=3),
            TokenSpec(name="surface_queries", size=4),
        ]
        patterns = [
            AttentionPattern(
                query_tokens=["surface_anchors", "surface_queries"],
                key_value_tokens=["surface_anchors"],
            )
        ]

        pad1 = torch.zeros(1, 1, dim)
        pad2 = torch.zeros(1, 2, dim)
        x_batched = torch.cat(
            [
                torch.cat([anchors_0, queries_0], dim=1),  # item 0: a0 a1 a2 q0..q3
                torch.cat([anchors_1, pad1, queries_1], dim=1),  # item 1: a0 a1 PAD q0..q3
                torch.cat([anchors_2, pad2, queries_2], dim=1),  # item 2: a0 PAD PAD q0..q3
            ],
            dim=0,
        )
        attn_mask = torch.tensor(
            [
                [True, True, True, True, True, True, True],  # item 0: all real
                [True, True, False, True, True, True, True],  # item 1: anchor 2 padding
                [True, False, False, True, True, True, True],  # item 2: anchors 1-2 padding
            ]
        )

        out_batched = module(x_batched, token_specs_padded, patterns, key_padding_mask=attn_mask)

        # Reference: run each item individually without mask or padding
        out_item0 = module(torch.cat([anchors_0, queries_0], dim=1), token_specs_padded, patterns)

        token_specs_2anchors = [TokenSpec(name="surface_anchors", size=2), TokenSpec(name="surface_queries", size=4)]
        out_item1 = module(torch.cat([anchors_1, queries_1], dim=1), token_specs_2anchors, patterns)

        token_specs_1anchor = [TokenSpec(name="surface_anchors", size=1), TokenSpec(name="surface_queries", size=4)]
        out_item2 = module(torch.cat([anchors_2, queries_2], dim=1), token_specs_1anchor, patterns)

        # Item 0: all positions must match (no padding)
        assert torch.allclose(out_batched[0], out_item0[0], atol=1e-5)

        # Item 1: compare real positions only
        # Batched layout:  a0(0) a1(1) PAD(2) q0(3) q1(4) q2(5) q3(6)
        # Unpadded layout: a0(0) a1(1)        q0(2) q1(3) q2(4) q3(5)
        assert torch.allclose(out_batched[1, [0, 1]], out_item1[0, [0, 1]], atol=1e-5)
        assert torch.allclose(out_batched[1, [3, 4, 5, 6]], out_item1[0, [2, 3, 4, 5]], atol=1e-5)

        # Item 2: compare real positions only
        # Batched layout:  a0(0) PAD(1) PAD(2) q0(3) q1(4) q2(5) q3(6)
        # Unpadded layout: a0(0)               q0(1) q1(2) q2(3) q3(4)
        assert torch.allclose(out_batched[2, [0]], out_item2[0, [0]], atol=1e-5)
        assert torch.allclose(out_batched[2, [3, 4, 5, 6]], out_item2[0, [1, 2, 3, 4]], atol=1e-5)
