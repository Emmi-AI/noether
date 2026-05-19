#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.modeling.modules.layers import ScalarsConditioner
from noether.modeling.modules.layers.scalar_conditioner import ScalarsConditionerConfig


def test_single():
    torch.manual_seed(0)
    model = ScalarsConditioner(ScalarsConditionerConfig(hidden_dim=4, num_scalars=1, condition_dim=6))
    y1 = model(friction_angle=torch.tensor([4.3]))
    y2 = model(torch.tensor([4.3]))
    expected = torch.tensor([-1.1654e-06, 1.5757e-06, -1.4685e-05, 4.2270e-06, -7.8820e-07, 1.1358e-06])
    assert torch.allclose(y1.squeeze(0), expected)
    assert torch.allclose(y2.squeeze(0), expected)


def test_double_kwargs():
    torch.manual_seed(0)
    model = ScalarsConditioner(ScalarsConditionerConfig(hidden_dim=4, num_scalars=2, condition_dim=6))
    y1 = model(friction_angle=torch.tensor([4.3]), geometry_angle=torch.tensor([4.3]))
    y2 = model(torch.tensor([4.3]), torch.tensor([4.3]))
    expected = torch.tensor([-1.4762e-06, 4.2751e-07, -1.0098e-06, -9.7093e-07, 2.1846e-06, -1.2073e-06])
    assert torch.allclose(y1.squeeze(0), expected)
    assert torch.allclose(y2.squeeze(0), expected)


def test_raises_on_invalid_num_scalars():
    model = ScalarsConditioner(ScalarsConditionerConfig(hidden_dim=4, num_scalars=2))
    with pytest.raises(AssertionError):
        model(geometry_angle=torch.tensor([4.3]))
