#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from noether.core.models import ModelBase


@torch.no_grad()
def copy_params(source_model: ModelBase, target_model: ModelBase) -> None:
    """Copy parameters and buffers from source_model to target_model."""
    target_params: list[torch.Tensor] = list(target_model.parameters())
    source_params: list[torch.Tensor] = list(source_model.parameters())
    if target_params:
        torch._foreach_mul_(target_params, 0.0)
        torch._foreach_add_(target_params, source_params, alpha=1.0)

    for target_buffer, source_buffer in zip(target_model.buffers(), source_model.buffers(), strict=True):
        target_buffer.copy_(source_buffer)


def compute_model_norm(module: nn.Module) -> torch.Tensor:
    """Norm of all weights of a module (useful for init sanity checks)."""
    return sum(p.norm() for p in module.parameters())  # type: ignore[no-any-return]


@torch.no_grad()
def update_ema(
    source_model: ModelBase, target_model: ModelBase, target_factor: float, copy_buffers: bool = False
) -> None:
    """Update the target model with an exponential moving average of the source model.

    Args:
        source_model: The source model to copy parameters from.
        target_model: The target model to update.
        target_factor: The factor to use for the exponential moving average.
        copy_buffers: Whether to copy buffers as well. Defaults to False.
    """
    # basic inplace implementation
    # for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
    #     target_param.mul_(target_factor).add_(source_param, alpha=1. - target_factor)

    # fused inplace implementation
    target_param_list: list[torch.Tensor] = list(target_model.parameters())
    source_param_list: list[torch.Tensor] = list(source_model.parameters())
    if len(target_param_list) > 0:
        # noinspection PyProtectedMember
        torch._foreach_mul_(target_param_list, target_factor)
        # noinspection PyProtectedMember
        torch._foreach_add_(target_param_list, source_param_list, alpha=1 - target_factor)

    if copy_buffers:
        for target_buffer, source_buffer in zip(target_model.buffers(), source_model.buffers(), strict=True):
            target_buffer.copy_(source_buffer)
