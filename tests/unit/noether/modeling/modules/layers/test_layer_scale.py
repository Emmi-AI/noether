#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch
from torch import nn

from noether.modeling.modules.layers.layer_scale import LayerScale, LayerScaleConfig


def test_layer_scale_with_default_init_scale():
    dim = 4
    config = LayerScaleConfig(hidden_dim=dim)
    layer = LayerScale(config)
    x = torch.ones((2, dim))
    output = layer(x)
    assert output.shape == x.shape
    assert torch.allclose(output, x * 1e-5)


def test_layer_scale_with_custom_init_scale():
    dim = 4
    init_scale = 0.1
    config = LayerScaleConfig(hidden_dim=dim, init_values=init_scale)
    layer = LayerScale(config)
    x = torch.ones((2, dim))
    output = layer(x)
    assert output.shape == x.shape
    assert torch.allclose(output, x * init_scale)


def test_layer_scale_with_none_init_scale():
    dim = 4
    config = LayerScaleConfig(hidden_dim=dim, init_values=None)
    layer = LayerScale(config)
    x = torch.ones((2, dim))
    output = layer(x)
    assert output.shape == x.shape
    assert torch.allclose(output, x)


def test_layer_scale_gamma_parameter():
    dim = 4
    init_scale = 0.1
    config = LayerScaleConfig(hidden_dim=dim, init_values=init_scale)
    layer = LayerScale(config)
    assert isinstance(layer.gamma, nn.Parameter)
    assert torch.allclose(layer.gamma, torch.full((dim,), init_scale))


def test_layer_scale_no_gamma_parameter():
    dim = 4
    config = LayerScaleConfig(hidden_dim=dim, init_values=None)
    layer = LayerScale(config)
    assert layer.gamma is None


def test_layer_scale_forward_backward():
    dim = 4
    init_scale = 1e-3
    config = LayerScaleConfig(hidden_dim=dim, init_values=init_scale)
    layer = LayerScale(config)
    x = torch.ones((2, dim), requires_grad=True)
    output = layer(x)
    output.sum().backward()
    assert output.shape == x.shape
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones((2, dim)) * init_scale)
