#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch
from torch import nn

ALL_CONVS = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
)

ALL_LAYERS = (
    nn.Linear,
    *ALL_CONVS,
)


def init_bias_to_zero(layer_module: nn.Module) -> None:
    """Initialize the bias tensor of a nn.Module instance to zero.

    Args:
        layer_module: An nn.Module instance, either a Linear or Conv layer.
    """
    if isinstance(layer_module, ALL_LAYERS):
        if layer_module.bias is not None:
            nn.init.constant_(layer_module.bias, 0.0)


def init_trunc_normal_zero_bias(layer_module: nn.Module, std: float = 0.02) -> None:
    """Initialize the weight tensor of a nn.Module instance using the truncated normal initialization with a zero bias
    vector.

    Args:
        layer_module: An nn.Module instance, either a Linear or Conv layer.
        std: Standard Deviation value of the normal distribution to sample weights from. Defaults to 0.02.
    """
    if isinstance(layer_module, ALL_LAYERS):
        nn.init.trunc_normal_(layer_module.weight, std=std)
        if layer_module.bias is not None:
            nn.init.constant_(layer_module.bias, 0.0)


def init_xavier_uniform_zero_bias(layer_module: nn.Module) -> None:
    """Initialize the weight tensor of a nn.Module instance using Xavier uniform with a zero bias vector.

    Args:
        layer_module: An nn.Module instance, either a Linear or Conv layer.
    """
    if isinstance(layer_module, ALL_LAYERS):
        nn.init.xavier_uniform_(layer_module.weight)
        if layer_module.bias is not None:
            nn.init.constant_(layer_module.bias, 0.0)


def apply_init_method(
    module: torch.nn.Module,
    proj_weight: torch.Tensor,
    init_method: str,
) -> None:
    """Apply an initialization function to all applicable sub-modules of a given module.

    Args:
        module: The nn.Module instance to initialize.
        init_fn: The initialization function to apply to each sub-module.
    """
    if init_method == "torch":
        pass
    elif init_method in ["truncnormal", "truncnormal002"]:
        module.apply(init_trunc_normal_zero_bias)
    elif init_method == "truncnormal002-identity":
        module.apply(init_trunc_normal_zero_bias)
        torch.nn.init.zeros_(proj_weight)
    elif init_method == "xavier":
        module.apply(init_xavier_uniform_zero_bias)
    else:
        raise NotImplementedError(f"Weight initialization method {init_method} not implemented for DotProductAttention")
