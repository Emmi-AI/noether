#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import math

import pytest
import torch

from noether.core.schemas.modules.layers import ContinuousSincosEmbeddingConfig
from noether.modeling.modules.layers.continuous_sincos_embed import ContinuousSincosEmbed

from .expected_output import CONTINUOUS_SINCOS_EMBED


def test_continuous_sincos_embed_init_valid():
    dim = 128
    ndim = 3
    max_wavelength = 10000

    config = ContinuousSincosEmbeddingConfig(hidden_dim=dim, input_dim=ndim, max_wavelength=max_wavelength)
    embed = ContinuousSincosEmbed(config)

    assert embed.hidden_dim == dim
    assert embed.omega.shape[0] == 21
    assert embed.input_dim == ndim
    assert embed.max_wavelength == max_wavelength
    assert embed.padding == (dim % ndim) + (((dim - (dim % ndim)) // ndim) % 2) * ndim
    assert embed.omega is not None


def test_continuous_sincos_embed_init_invalid_dim():
    dim = 3
    ndim = 4

    with pytest.raises(AssertionError):
        config = ContinuousSincosEmbeddingConfig(hidden_dim=dim, input_dim=ndim)
        ContinuousSincosEmbed(config)


def test_continous_sincos_embed_forward():
    torch.manual_seed(42)
    dim = 16
    ndim = 3
    num_coords = 8
    config = ContinuousSincosEmbeddingConfig(hidden_dim=dim, input_dim=ndim)
    embed = ContinuousSincosEmbed(config)

    coords = torch.rand(2, num_coords, ndim)
    out = embed(coords)

    assert out.shape == (2, num_coords, dim)
    assert torch.allclose(out, CONTINUOUS_SINCOS_EMBED, rtol=1e-4)


def test_continous_sincos_embed_forward_fp32forced():
    torch.manual_seed(42)
    dim = 16
    ndim = 3
    num_coords = 8
    config = ContinuousSincosEmbeddingConfig(hidden_dim=dim, input_dim=ndim)
    embed = ContinuousSincosEmbed(config)

    coords = torch.rand(2, num_coords, ndim)
    with torch.autocast(device_type="cpu", dtype=torch.float16):
        out_fp16 = embed(coords)
    out_fp32 = embed(coords)
    assert torch.equal(out_fp16, out_fp32)


def test_continuous_sincos_embed_nerf_omega_endpoints():
    dim = 192
    ndim = 3
    max_frequency = 32.0
    config = ContinuousSincosEmbeddingConfig(hidden_dim=dim, input_dim=ndim, mode="nerf", max_frequency=max_frequency)
    embed = ContinuousSincosEmbed(config)

    # L = (192 / 3) / 2 = 32 bands, log-spaced between π and π*max_frequency.
    assert embed.omega.shape[0] == 32
    assert torch.isclose(embed.omega[0], torch.tensor(math.pi))
    assert torch.isclose(embed.omega[-1], torch.tensor(math.pi * max_frequency))


def test_continuous_sincos_embed_nerf_resolves_fine_scales():
    """The motivating benefit of NeRF mode: it resolves coordinate differences much
    smaller than wavelength mode can, since wavelength mode's highest frequency is
    omega=1 (period 2π) while NeRF mode reaches omega=π * max_frequency."""
    dim = 64
    ndim = 3
    delta = 0.01
    coords = torch.tensor([[0.0, 0.0, 0.0], [delta, delta, delta]])

    wavelength_embed = ContinuousSincosEmbed(ContinuousSincosEmbeddingConfig(hidden_dim=dim, input_dim=ndim))
    nerf_embed = ContinuousSincosEmbed(
        ContinuousSincosEmbeddingConfig(hidden_dim=dim, input_dim=ndim, mode="nerf", max_frequency=32.0)
    )

    wavelength_diff = (wavelength_embed(coords)[0] - wavelength_embed(coords)[1]).abs().max()
    nerf_diff = (nerf_embed(coords)[0] - nerf_embed(coords)[1]).abs().max()

    assert nerf_diff > 0.5
    assert nerf_diff > 10 * wavelength_diff


def test_continuous_sincos_embed_nerf_zero_input():
    dim = 60  # cleanly divisible: 60 / 3 = 20, no padding
    ndim = 3
    config = ContinuousSincosEmbeddingConfig(hidden_dim=dim, input_dim=ndim, mode="nerf", max_frequency=32.0)
    embed = ContinuousSincosEmbed(config)
    assert embed.padding == 0

    out = embed(torch.zeros(1, ndim))
    # Per-axis layout is [sin(0)*L, cos(0)*L].
    L = embed.omega.shape[0]
    out_per_axis = out.reshape(ndim, 2 * L)
    assert torch.allclose(out_per_axis[:, :L], torch.zeros_like(out_per_axis[:, :L]))
    assert torch.allclose(out_per_axis[:, L:], torch.ones_like(out_per_axis[:, L:]))


def test_continuous_sincos_embed_nerf_requires_max_frequency():
    with pytest.raises(ValueError, match="max_frequency"):
        ContinuousSincosEmbeddingConfig(hidden_dim=64, input_dim=3, mode="nerf")
