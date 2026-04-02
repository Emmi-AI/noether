#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.core.schemas.modules.blocks import PerceiverBlockConfig, TransformerBlockConfig
from noether.core.schemas.modules.layers import RopeFrequencyConfig
from noether.modeling.functional.rope import rope
from noether.modeling.modules.blocks import PerceiverBlock, TransformerBlock
from noether.modeling.modules.layers import RopeFrequency

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

EXPECTED_OUTPUT_TRANSFORMER = torch.Tensor(
    [
        [
            [
                1.2016e00,
                3.4670e-01,
                9.6649e-01,
                -6.2874e-01,
                1.0647e00,
                -1.8693e00,
                -1.1490e00,
                2.0133e00,
                -1.3319e00,
                4.9961e-01,
                -1.3232e-01,
                -1.0726e-01,
                -6.0204e-01,
                -1.3064e00,
                6.9178e-01,
                -1.0479e00,
            ],
            [
                -1.1403e-01,
                1.1538e00,
                -4.4908e-01,
                1.2045e00,
                8.1183e-01,
                2.1009e00,
                1.1232e00,
                1.3549e00,
                -5.1542e-01,
                -3.3760e-01,
                -9.5836e-01,
                1.2702e-02,
                4.7730e-01,
                3.6324e-01,
                7.4763e-02,
                7.8179e-01,
            ],
            [
                1.0810e00,
                1.9360e00,
                5.4689e-01,
                4.3009e-02,
                -6.4128e-01,
                1.9479e00,
                9.0156e-01,
                -8.9924e-01,
                -4.3067e-01,
                1.8797e00,
                2.7805e-01,
                1.5734e00,
                -4.8331e-01,
                -7.5125e-01,
                -1.6322e00,
                -1.3513e00,
            ],
        ],
        [
            [
                -1.0274e00,
                -6.4449e-01,
                -3.1495e00,
                1.7645e-03,
                5.7879e-01,
                1.7817e00,
                -7.3272e-01,
                8.1544e-01,
                5.5103e-02,
                2.4562e-01,
                -1.3593e00,
                3.4795e-01,
                -3.4434e-02,
                -8.0227e-03,
                1.9691e00,
                -1.1701e00,
            ],
            [
                4.4651e-01,
                2.7101e-01,
                -1.1847e00,
                -9.9352e-01,
                -1.1059e00,
                -9.2329e-01,
                9.9352e-01,
                -4.0599e-01,
                -8.6103e-01,
                -4.7311e-01,
                -5.7329e-01,
                2.7870e-01,
                1.9281e00,
                -8.2982e-01,
                -1.3963e-01,
                2.2341e-01,
            ],
            [
                -2.7558e-01,
                -1.4914e-01,
                -8.3003e-01,
                -3.1375e-01,
                -9.8371e-02,
                -2.1650e-01,
                -2.2605e00,
                4.5438e-01,
                1.2300e00,
                1.5715e-01,
                -1.1405e00,
                -7.7377e-01,
                9.6159e-01,
                -7.0842e-01,
                -1.2350e00,
                -1.8665e00,
            ],
        ],
    ]
)

EXPECTED_OUTPUT_PERCEIVER = torch.Tensor(
    [
        [
            [
                1.2018,
                0.3431,
                0.9595,
                -0.6259,
                1.0620,
                -1.8655,
                -1.1567,
                2.0177,
                -1.3313,
                0.5038,
                -0.1180,
                -0.1093,
                -0.6033,
                -1.2977,
                0.6872,
                -1.0491,
            ],
            [
                -0.1138,
                1.1502,
                -0.4561,
                1.2073,
                0.8091,
                2.1046,
                1.1155,
                1.3594,
                -0.5148,
                -0.3334,
                -0.9440,
                0.0107,
                0.4761,
                0.3719,
                0.0701,
                0.7806,
            ],
            [
                1.0812,
                1.9324,
                0.5399,
                0.0458,
                -0.6440,
                1.9517,
                0.8939,
                -0.8948,
                -0.4301,
                1.8839,
                0.2924,
                1.5714,
                -0.4845,
                -0.7425,
                -1.6368,
                -1.3525,
            ],
        ],
        [
            [
                -1.0268,
                -0.6347,
                -3.1507,
                0.0095,
                0.5852,
                1.7874,
                -0.7276,
                0.8102,
                0.0664,
                0.2419,
                -1.3618,
                0.3388,
                -0.0331,
                -0.0114,
                1.9774,
                -1.1582,
            ],
            [
                0.4470,
                0.2808,
                -1.1859,
                -0.9858,
                -1.0995,
                -0.9176,
                0.9986,
                -0.4112,
                -0.8497,
                -0.4769,
                -0.5758,
                0.2696,
                1.9295,
                -0.8332,
                -0.1313,
                0.2354,
            ],
            [
                -0.2751,
                -0.1393,
                -0.8313,
                -0.3060,
                -0.0920,
                -0.2107,
                -2.2554,
                0.4492,
                1.2413,
                0.1534,
                -1.1430,
                -0.7829,
                0.9630,
                -0.7117,
                -1.2266,
                -1.8546,
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
