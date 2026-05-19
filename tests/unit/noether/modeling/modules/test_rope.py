#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.modeling.functional.rope import rope
from noether.modeling.modules.blocks import PerceiverBlock, TransformerBlock
from noether.modeling.modules.blocks.perceiver import PerceiverBlockConfig
from noether.modeling.modules.blocks.transformer import TransformerBlockConfig
from noether.modeling.modules.layers import RopeFrequency
from noether.modeling.modules.layers.rope_frequency import RopeFrequencyConfig

EXPECTED_OUTPUT_STANDALONE = torch.Tensor(
    [
        [
            [
                [-0.4413, -1.5494, 0.1213, -0.4861, 0.7842, 0.7643, -0.3160, -2.1152],
                [0.4858, -1.2099, 0.2403, 0.3996, -0.6367, 1.0681, 1.1168, -0.2473],
                [-0.3951, -2.1330, -0.2660, 0.9381, 1.2220, -1.1329, -0.3414, 1.8530],
            ],
            [
                [0.9385, -0.1577, -0.2522, 0.0115, 1.2438, 1.7029, 0.9463, -0.8437],
                [-0.6124, -0.0495, -0.5447, 0.0877, 0.2876, 0.3511, 0.6408, 0.4412],
                [-0.4633, 0.6510, -0.2219, -0.1935, -0.5435, 2.2974, -1.4689, -1.5867],
            ],
        ],
    ]
)

EXPECTED_OUTPUT_TRANSFORMER = torch.tensor(
    [
        [
            [
                1.2020e00,
                3.4760e-01,
                9.6740e-01,
                -6.2807e-01,
                1.0641e00,
                -1.8679e00,
                -1.1471e00,
                2.0125e00,
                -1.3303e00,
                4.9845e-01,
                -1.3191e-01,
                -1.0934e-01,
                -6.0162e-01,
                -1.3069e00,
                6.9048e-01,
                -1.0482e00,
            ],
            [
                -1.1762e-01,
                1.1533e00,
                -4.5457e-01,
                1.2040e00,
                8.1329e-01,
                2.0946e00,
                1.1236e00,
                1.3552e00,
                -5.1526e-01,
                -3.3662e-01,
                -9.5354e-01,
                2.1194e-02,
                4.7831e-01,
                3.6832e-01,
                7.8824e-02,
                7.8473e-01,
            ],
            [
                1.0792e00,
                1.9358e00,
                5.4524e-01,
                4.3306e-02,
                -6.4067e-01,
                1.9454e00,
                9.0167e-01,
                -8.9915e-01,
                -4.2944e-01,
                1.8791e00,
                2.8104e-01,
                1.5766e00,
                -4.8257e-01,
                -7.4883e-01,
                -1.6302e00,
                -1.3503e00,
            ],
        ],
        [
            [
                -1.0259e00,
                -6.4528e-01,
                -3.1489e00,
                2.4705e-03,
                5.7916e-01,
                1.7831e00,
                -7.3340e-01,
                8.1630e-01,
                5.3415e-02,
                2.4654e-01,
                -1.3614e00,
                3.4650e-01,
                -3.4810e-02,
                -8.6258e-03,
                1.9686e00,
                -1.1694e00,
            ],
            [
                4.4940e-01,
                2.7076e-01,
                -1.1832e00,
                -9.9185e-01,
                -1.1059e00,
                -9.2032e-01,
                9.9409e-01,
                -4.0536e-01,
                -8.6231e-01,
                -4.7268e-01,
                -5.7649e-01,
                2.7493e-01,
                1.9271e00,
                -8.3155e-01,
                -1.4182e-01,
                2.2350e-01,
            ],
            [
                -2.7133e-01,
                -1.4920e-01,
                -8.2463e-01,
                -3.1234e-01,
                -9.8254e-02,
                -2.1185e-01,
                -2.2583e00,
                4.5365e-01,
                1.2284e00,
                1.5603e-01,
                -1.1457e00,
                -7.8025e-01,
                9.6168e-01,
                -7.1040e-01,
                -1.2397e00,
                -1.8677e00,
            ],
        ],
    ]
)

EXPECTED_OUTPUT_PERCEIVER = torch.tensor(
    [
        [
            [
                1.2001e00,
                3.4561e-01,
                9.6043e-01,
                -6.2427e-01,
                1.0608e00,
                -1.8634e00,
                -1.1521e00,
                2.0166e00,
                -1.3288e00,
                5.0434e-01,
                -1.1973e-01,
                -1.1178e-01,
                -6.0135e-01,
                -1.3007e00,
                6.8579e-01,
                -1.0478e00,
            ],
            [
                -1.1953e-01,
                1.1513e00,
                -4.6157e-01,
                1.2077e00,
                8.1007e-01,
                2.0992e00,
                1.1186e00,
                1.3593e00,
                -5.1379e-01,
                -3.3072e-01,
                -9.4129e-01,
                1.8699e-02,
                4.7863e-01,
                3.7455e-01,
                7.4125e-02,
                7.8516e-01,
            ],
            [
                1.0774e00,
                1.9338e00,
                5.3827e-01,
                4.7093e-02,
                -6.4390e-01,
                1.9500e00,
                8.9662e-01,
                -8.9505e-01,
                -4.2797e-01,
                1.8850e00,
                2.9324e-01,
                1.5742e00,
                -4.8229e-01,
                -7.4257e-01,
                -1.6349e00,
                -1.3499e00,
            ],
        ],
        [
            [
                -1.0249e00,
                -6.3701e-01,
                -3.1494e00,
                1.0203e-02,
                5.8396e-01,
                1.7887e00,
                -7.2731e-01,
                8.1116e-01,
                6.4484e-02,
                2.4214e-01,
                -1.3626e00,
                3.3919e-01,
                -3.3264e-02,
                -1.1320e-02,
                1.9753e00,
                -1.1594e00,
            ],
            [
                4.5034e-01,
                2.7893e-01,
                -1.1836e00,
                -9.8420e-01,
                -1.1011e00,
                -9.1470e-01,
                1.0000e00,
                -4.1049e-01,
                -8.5132e-01,
                -4.7707e-01,
                -5.7761e-01,
                2.6768e-01,
                1.9287e00,
                -8.3415e-01,
                -1.3512e-01,
                2.3356e-01,
            ],
            [
                -2.7040e-01,
                -1.4094e-01,
                -8.2514e-01,
                -3.0468e-01,
                -9.3499e-02,
                -2.0616e-01,
                -2.2523e00,
                4.4854e-01,
                1.2395e00,
                1.5168e-01,
                -1.1468e00,
                -7.8760e-01,
                9.6318e-01,
                -7.1299e-01,
                -1.2331e00,
                -1.8576e00,
            ],
        ],
    ]
)


@pytest.mark.parametrize("implementation", ["real", "complex"])
def test_rope_standalone(implementation):
    batch_size = 1
    num_heads = 2
    num_points = 3
    ndim = 3
    dim = 16
    assert dim % num_heads == 0
    head_dim = dim // num_heads
    x = torch.randn(batch_size, num_heads, num_points, head_dim, generator=torch.Generator().manual_seed(0))
    pos = torch.rand(batch_size, num_points, ndim, generator=torch.Generator().manual_seed(0))
    freqs = RopeFrequency(
        RopeFrequencyConfig(
            hidden_dim=head_dim,
            input_dim=ndim,
            max_wavelength=10000,
            implementation=implementation,
        )
    )(pos)
    y = rope(x=x, freqs=freqs)
    assert torch.allclose(y, EXPECTED_OUTPUT_STANDALONE, atol=1e-4)
    # head_dim=8 is not divisible by ndim * 2 -> frequencies for these dimensions are 0 -> not rotated
    assert torch.equal(x[:, :, ndim * 2 :], y[:, :, ndim * 2 :])


@pytest.mark.parametrize("implementation", ["real", "complex"])
def test_rope_transformer(implementation):
    torch.manual_seed(0)
    batch_size = 2
    num_points = 3
    ndim = 3
    dim = 16
    num_heads = 2
    block = TransformerBlock(
        TransformerBlockConfig(  # type: ignore
            hidden_dim=dim,
            num_heads=num_heads,
            mlp_expansion_factor=4,
            use_rope=True,
        )
    )
    x = torch.randn(batch_size, num_points, dim)
    pos = torch.rand(batch_size, num_points, ndim)
    freqs = RopeFrequency(
        RopeFrequencyConfig(
            hidden_dim=block.attention_block.head_dim,
            input_dim=ndim,
            max_wavelength=10000,
            implementation=implementation,
        )
    )(pos)
    y, _ = block(x, attn_kwargs=dict(freqs=freqs))
    assert torch.allclose(y, EXPECTED_OUTPUT_TRANSFORMER, atol=1e-4)


@pytest.mark.parametrize("implementation", ["real", "complex"])
def test_rope_transformer_bs_neq_numheads(implementation):
    torch.manual_seed(0)
    batch_size = 5
    num_points = 3
    ndim = 3
    dim = 16
    num_heads = 2
    block = TransformerBlock(
        TransformerBlockConfig(  # type: ignore
            hidden_dim=dim,
            num_heads=num_heads,
            mlp_expansion_factor=4,
            use_rope=True,
        )
    )
    x = torch.randn(batch_size, num_points, dim)
    pos = torch.rand(batch_size, num_points, ndim)
    freqs = RopeFrequency(
        RopeFrequencyConfig(
            hidden_dim=block.attention_block.head_dim,
            input_dim=ndim,
            max_wavelength=10000,
            implementation=implementation,
        )
    )(pos)
    block(x, attn_kwargs=dict(freqs=freqs))


@pytest.mark.parametrize("implementation", ["real", "complex"])
def test_rope_perceiver(implementation):
    torch.manual_seed(0)
    batch_size = 2
    q_num_points = 3
    kv_num_points = 2
    ndim = 3
    dim = 16
    num_heads = 2
    head_dim = dim // num_heads
    block = PerceiverBlock(
        config=PerceiverBlockConfig(  # type: ignore
            hidden_dim=dim,
            num_heads=num_heads,
            mlp_expansion_factor=4,
            use_rope=True,
        )
    )
    q = torch.randn(batch_size, q_num_points, dim)
    kv = torch.randn(batch_size, kv_num_points, dim)
    q_pos = torch.rand(batch_size, q_num_points, ndim)
    kv_pos = torch.rand(batch_size, kv_num_points, ndim)
    rope_freqs = RopeFrequency(
        RopeFrequencyConfig(
            hidden_dim=head_dim,
            input_dim=ndim,
            max_wavelength=10000,
            implementation=implementation,
        )
    )
    q_freqs = rope_freqs(q_pos)
    k_freqs = rope_freqs(kv_pos)
    y, _ = block(q=q, kv=kv, attn_kwargs=dict(q_freqs=q_freqs, k_freqs=k_freqs))
    assert torch.allclose(y, EXPECTED_OUTPUT_PERCEIVER, atol=1e-4)
