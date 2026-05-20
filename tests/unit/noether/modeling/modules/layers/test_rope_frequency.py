#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch

from noether.modeling.modules.layers import RopeFrequency
from noether.modeling.modules.layers.rope_frequency import RopeFrequencyConfig


def test_3d():
    ndim = 3
    pos = torch.rand(1, ndim, generator=torch.Generator().manual_seed(0))
    config = RopeFrequencyConfig(hidden_dim=8, input_dim=ndim, implementation="real")
    rope = RopeFrequency(config)
    freqs = rope(pos)
    assert len(freqs) == 3
    assert torch.is_tensor(freqs[0]) and freqs[0].shape == (1, 2)
    assert torch.is_tensor(freqs[1]) and freqs[1].shape == (1, 2)
    assert torch.is_tensor(freqs[2]) and freqs[2].shape == (1, 2)
