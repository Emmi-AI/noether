#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.modeling.modules.mlp.upactdown_mlp import UpActDownMlp, UpActDownMLPConfig

from .expected_output import UPACTDOWN_MLP


@pytest.mark.parametrize(
    ("input_dim", "hidden_dim", "init_weights"),
    [
        (4, 8, "torch"),
        (4, 8, "truncnormal002"),
    ],
)
def test_upactdown_mlp_forward(input_dim, hidden_dim, init_weights):
    config = UpActDownMLPConfig(input_dim=input_dim, hidden_dim=hidden_dim, init_weights=init_weights)
    model = UpActDownMlp(config)
    x = torch.randn(2, input_dim)  # Batch size of 2
    output = model(x)
    assert output.shape == (2, input_dim), "Output shape mismatch"


def test_upactdown_mlp_invalid_dim_inputs():
    with pytest.raises(ValueError):
        config = UpActDownMLPConfig(input_dim=4, hidden_dim=0)

    with pytest.raises(ValueError):
        config = UpActDownMLPConfig(input_dim=8, hidden_dim=4)

    with pytest.raises(ValueError):
        config = UpActDownMLPConfig(input_dim=0, hidden_dim=2)


def test_upactdown_mlp_invalid_init_weights():
    from pydantic_core import ValidationError

    with pytest.raises(ValidationError):
        config = UpActDownMLPConfig(input_dim=4, hidden_dim=8, init_weights="invalid")


def test_upactdown_mlp_reset_parameters():
    config = UpActDownMLPConfig(input_dim=4, hidden_dim=8, init_weights="torch")
    model = UpActDownMlp(config)
    # Ensure reset_parameters does not raise errors
    fc1 = model.fc1.weight.clone()
    model.reset_parameters()
    assert torch.allclose(model.fc1.weight, fc1), "Weights should not change"

    config = UpActDownMLPConfig(input_dim=4, hidden_dim=8, init_weights="truncnormal002")
    model = UpActDownMlp(config)
    fc1 = model.fc1.weight.clone()
    model.reset_parameters()
    assert torch.all(model.fc1.weight != fc1), "Weights should change"


def test_upactdown_mlp_forward_backward():
    torch.manual_seed(42)
    config = UpActDownMLPConfig(input_dim=4, hidden_dim=8, init_weights="torch")
    model = UpActDownMlp(config)
    x = torch.randn(2, 4)  # Batch size of 2
    output = model(x)

    output.sum().backward()

    assert torch.allclose(output, UPACTDOWN_MLP, 1e-2), "Output value mismatch"
    assert output.shape == (2, 4), "Output shape mismatch"
    assert model.fc1.weight.grad is not None, "Gradients should not be None"


def test_no_bias():
    config = UpActDownMLPConfig(input_dim=4, hidden_dim=8, bias=False)
    mlp = UpActDownMlp(config)
    assert mlp.fc1.bias is None
    assert mlp.fc2.bias is None


def test_truncnormal_init0():
    config = UpActDownMLPConfig(input_dim=4, hidden_dim=8, init_weights="truncnormal002-identity")
    mlp = UpActDownMlp(config)
    assert torch.all(mlp.fc2.weight == 0)
    assert torch.all(mlp.fc2.bias == 0)
