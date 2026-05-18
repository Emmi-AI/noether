#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.core.schemas.models import ViTConfig
from noether.modeling.models import ViT

_HIDDEN_DIM = 32


def _make_inputs(B: int = 2, H: int = 16, W: int = 16, coord_dim: int = 2):
    return dict(
        x=None,
        coords=torch.randn(B, H, W, coord_dim),
        mask=torch.ones(B, H, W, dtype=torch.bool),
        cond=torch.randn(B, _HIDDEN_DIM),
    )


def _make_model(**overrides):
    cfg = dict(
        name="vit_test",
        coord_dim=2,
        out_channels=4,
        patch_size=8,
        hidden_dim=_HIDDEN_DIM,
        num_heads=2,
        depth=2,
    )
    cfg.update(overrides)
    return ViT(config=ViTConfig(**cfg))


def test_forward_shape_conv_head():
    model = _make_model()
    out = model(**_make_inputs())
    assert out.shape == (2, 16, 16, 4)


def test_forward_shape_linear_unpatchify():
    model = _make_model(use_conv_output_head=False)
    out = model(**_make_inputs())
    assert out.shape == (2, 16, 16, 4)


def test_forward_without_mask():
    model = _make_model()
    inputs = _make_inputs()
    inputs["mask"] = None
    out = model(**inputs)
    assert out.shape == (2, 16, 16, 4)


def test_fully_solid_mask_does_not_nan():
    model = _make_model().eval()
    inputs = _make_inputs()
    inputs["mask"] = torch.zeros_like(inputs["mask"])
    out = model(**inputs)
    assert not torch.isnan(out).any()


def test_return_tokens():
    """With use_conv_output_head=True, FinalLayer is configured with patch_size=1 and
    out_channels=hidden_dim, so its output is (B, num_patches, hidden_dim)."""
    model = _make_model()
    tokens, (gh, gw) = model(**_make_inputs(), return_tokens=True)
    assert (gh, gw) == (2, 2)  # H=16, patch=8 => 2x2 patches
    assert tokens.shape == (2, gh * gw, 32)


def test_cond_required():
    model = _make_model()
    inputs = _make_inputs()
    inputs["cond"] = None
    with pytest.raises(ValueError, match="cond"):
        model(**inputs)


def test_arbitrary_grid_shape():
    """The ViT works on rectangular grids as long as patch_size divides H and W."""
    model = _make_model(patch_size=4)
    coords = torch.randn(1, 32, 16, 2)
    out = model(x=None, coords=coords, cond=torch.randn(1, _HIDDEN_DIM))
    assert out.shape == (1, 32, 16, 4)


def test_unconditioned_vit():
    """With ``use_conditioning=False`` the ViT runs without ``cond`` and has no AdaLN machinery."""
    model = _make_model(use_conditioning=False)
    # No per-block AdaLN modulation submodule.
    for block in model.backbone.blocks:
        assert block.modulation is None
    # FinalLayer has no AdaLN modulation submodule either.
    assert model.final_layer.adaLN_modulation is None

    inputs = _make_inputs()
    inputs["cond"] = None
    out = model(**inputs)
    assert out.shape == (2, 16, 16, 4)

    # Passing cond when conditioning is disabled is an error.
    inputs["cond"] = torch.randn(2, _HIDDEN_DIM)
    with pytest.raises(ValueError, match="must be None"):
        model(**inputs)


def test_unconditioned_vit_backward_pass():
    model = _make_model(use_conditioning=False)
    inputs = _make_inputs()
    inputs["cond"] = None
    out = model(**inputs)
    out.sum().backward()
    grad = model.final_layer.linear.weight.grad
    # final_layer.linear is zero-initialized; gradient still flows through it.
    assert grad is not None
    assert not torch.isnan(grad).any()


def test_backward_pass_grads_first_block_modulation():
    """Drops the earlier cond_embedder-grad check: cond_embedder is gone. Verify a representative
    learnable parameter (the first block's AdaLN modulation Linear) gets a gradient instead."""
    model = _make_model()
    out = model(**_make_inputs())
    out.sum().backward()
    grad = model.backbone.blocks[0].modulation.project.weight.grad  # type: ignore[union-attr]
    assert grad is not None
    assert not torch.isnan(grad).any()
