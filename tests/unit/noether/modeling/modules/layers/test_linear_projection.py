#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.modeling.modules.layers.linear_projection import LinearProjection, LinearProjectionConfig

from .expected_output import LINEAR_PROJECTION


@pytest.mark.parametrize(
    ("input_dim", "output_dim", "ndim", "bias", "optional", "init_weights"),
    [
        (4, 4, None, True, True, "torch"),  # Identity case
        (4, 8, None, True, False, "torch"),  # Linear layer
        (4, 8, 1, True, False, "torch"),  # Conv1d
        (4, 8, 2, True, False, "torch"),  # Conv2d
        (4, 8, 3, True, False, "torch"),  # Conv3d
        (4, 8, None, True, False, "truncnormal"),  # Truncated normal init
    ],
)
def test_linear_projection_forward(input_dim, output_dim, ndim, bias, optional, init_weights):
    batch_size = 2
    if ndim is None:
        input_tensor = torch.randn(batch_size, input_dim)
    elif ndim == 1:
        input_tensor = torch.randn(batch_size, input_dim, 10)
    elif ndim == 2:
        input_tensor = torch.randn(batch_size, input_dim, 10, 10)
    elif ndim == 3:
        input_tensor = torch.randn(batch_size, input_dim, 10, 10, 10)
    else:
        pytest.fail("Unsupported ndim value")

    config = LinearProjectionConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        ndim=ndim,
        bias=bias,
        optional=optional,
        init_weights=init_weights,
    )
    model = LinearProjection(config)
    output = model(input_tensor)

    if optional and input_dim == output_dim:
        assert torch.equal(input_tensor, output), "Identity mapping failed"
    else:
        assert output.shape[1] == output_dim, "Output dimension mismatch"


def test_linear_projection_forward_backward():
    torch.manual_seed(42)
    input_dim = 4
    output_dim = 8
    ndim = None
    bias = True
    optional = False
    init_weights = "torch"

    batch_size = 2
    input_tensor = torch.randn(batch_size, input_dim)

    config = LinearProjectionConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        ndim=ndim,
        bias=bias,
        optional=optional,
        init_weights=init_weights,
    )
    model = LinearProjection(config)
    output = model(input_tensor)
    output.sum().backward()

    assert output.shape[1] == output_dim, "Output dimension mismatch"
    assert model.project.weight.grad is not None, "Gradients should not be None"
    assert torch.allclose(output, LINEAR_PROJECTION, 1e-2), "Output value mismatch"


def test_linear_projection_zeroinit():
    config = LinearProjectionConfig(input_dim=4, output_dim=8, init_weights="zeros")
    proj = LinearProjection(config)
    out = proj(torch.randn(1, 4))
    assert torch.allclose(out, torch.zeros(1, 8))


def test_linear_projection_invalid_ndim():
    with pytest.raises(ValueError):
        config = LinearProjectionConfig(input_dim=4, output_dim=8, ndim=4)


def test_linear_projection_invalid_init_weights():
    from pydantic_core import ValidationError

    with pytest.raises(ValidationError):
        config = LinearProjectionConfig(input_dim=4, output_dim=8, init_weights="invalid")
