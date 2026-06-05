#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.core.schemas.modules.attention import FLAREAttentionConfig, TokenSpec
from noether.core.schemas.modules.blocks import TransformerBlockConfig
from noether.modeling.modules.attention.flare import FLAREAttention
from noether.modeling.modules.blocks import TransformerBlock


def _config(**kwargs) -> FLAREAttentionConfig:
    return FLAREAttentionConfig(
        hidden_dim=16,
        num_heads=4,
        num_latents=5,
        num_layers_k_proj=-1,
        num_layers_v_proj=-1,
        **kwargs,
    )


def test_flare_attention_init() -> None:
    config = _config(dropout=0.1, bias=False, qk_norm=True)
    model = FLAREAttention(config)

    assert model.num_heads == 4
    assert model.head_dim == 4
    assert model.num_latents == 5
    assert model.dropout == 0.1
    assert model.latent_q.shape == (16, 5)
    assert model.k_proj.fc.bias is None
    assert model.v_proj.fc.bias is None
    assert isinstance(model.q_norm, torch.nn.LayerNorm)
    assert isinstance(model.k_norm, torch.nn.LayerNorm)


def test_flare_attention_forward_and_backward() -> None:
    torch.manual_seed(42)
    model = FLAREAttention(_config())
    x = torch.randn(2, 7, 16)

    out = model(x)

    assert out.shape == x.shape
    out.sum().backward()
    assert model.latent_q.grad is not None
    assert model.k_proj.fc.weight.grad is not None
    assert model.v_proj.fc.weight.grad is not None
    assert model.out_proj.weight.grad is not None


def test_flare_attention_mask_excludes_padding_from_real_outputs() -> None:
    torch.manual_seed(42)
    model = FLAREAttention(_config()).eval()
    x = torch.randn(2, 8, 16)
    x_alt = x.clone()
    x_alt[:, 4:] = torch.randn_like(x_alt[:, 4:]) * 100.0
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, :4] = True

    with torch.no_grad():
        out = model(x, key_padding_mask=mask)
        out_alt = model(x_alt, key_padding_mask=mask)

    assert torch.allclose(out[:, :4], out_alt[:, :4], atol=1e-5)


def test_flare_attention_token_specs_encode_only_anchors() -> None:
    torch.manual_seed(42)
    model = FLAREAttention(_config()).eval()
    x = torch.randn(2, 6, 16)
    x_alt = x.clone()
    x_alt[:, 3:] = torch.randn_like(x_alt[:, 3:]) * 100.0
    token_specs = [
        TokenSpec(name="reservoir_anchors", size=3),
        TokenSpec(name="reservoir_queries", size=3),
    ]

    with torch.no_grad():
        out = model(x, token_specs=token_specs)
        out_alt = model(x_alt, token_specs=token_specs)

    assert torch.allclose(out[:, :3], out_alt[:, :3], atol=1e-5)


def test_flare_attention_rejects_bad_mask_shape() -> None:
    model = FLAREAttention(_config())
    x = torch.randn(2, 7, 16)

    with pytest.raises(ValueError, match="token mask"):
        model(x, key_padding_mask=torch.ones(2, 6, dtype=torch.bool))


def test_transformer_block_registry_constructs_flare_attention() -> None:
    block = TransformerBlock(
        TransformerBlockConfig(
            hidden_dim=16,
            num_heads=4,
            mlp_expansion_factor=2,
            attention_constructor="flare",
            attention_arguments={
                "num_latents": 5,
                "num_layers_k_proj": -1,
                "num_layers_v_proj": -1,
            },
        )
    )
    x = torch.randn(2, 7, 16)

    out, kv_cache = block(x)

    assert out.shape == x.shape
    assert kv_cache is None
