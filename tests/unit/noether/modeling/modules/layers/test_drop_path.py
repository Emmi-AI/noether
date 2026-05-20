#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.modeling.modules.layers.drop_path import UnquantizedDropPath, UnquantizedDropPathConfig


@pytest.mark.parametrize(
    ("drop_prob", "scale_by_keep"),
    [
        (0.0, True),
        (0.0, False),
        (0.5, True),
        (0.5, False),
        (1.0, True),
        (1.0, False),
    ],
)
def test_unquantized_drop_path_forward(drop_prob, scale_by_keep):
    config = UnquantizedDropPathConfig(drop_prob=drop_prob, scale_by_keep=scale_by_keep)
    module = UnquantizedDropPath(config)
    module.train()  # Set to training mode

    x = torch.ones((4, 3, 32, 32))  # Example input tensor
    output = module(x)

    if drop_prob == 0.0:
        # If drop_prob is 0, output should be the same as input
        assert torch.equal(output, x)
    elif drop_prob == 1.0:
        # If drop_prob is 1, all paths should be dropped (output should be all zeros)
        assert torch.sum(output) == 0
    else:
        # For other drop_prob values, ensure output shape matches input shape
        assert output.shape == x.shape


def test_unquantized_drop_path_eval_mode():
    config = UnquantizedDropPathConfig(drop_prob=0.5, scale_by_keep=True)
    module = UnquantizedDropPath(config)
    module.eval()  # Set to evaluation mode

    x = torch.ones((4, 3, 32, 32))  # Example input tensor
    output = module(x)

    # In evaluation mode, output should be the same as input
    assert torch.equal(output, x)


def test_keep_prob_property():
    config = UnquantizedDropPathConfig(drop_prob=0.3)
    module = UnquantizedDropPath(config)
    assert module.keep_prob == 0.7  # keep_prob should be 1 - drop_prob


def test_extra_repr():
    config = UnquantizedDropPathConfig(drop_prob=0.3)
    module = UnquantizedDropPath(config)
    assert "drop_prob=0.30" in module.extra_repr()


def test_unvalid_drop_probs():
    with pytest.raises(ValueError):
        config = UnquantizedDropPathConfig(drop_prob=-0.1)
    with pytest.raises(ValueError):
        config = UnquantizedDropPathConfig(drop_prob=1.1)
