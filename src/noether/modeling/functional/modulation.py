#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch


def modulate_scale_shift(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Scales and shifts the input x featurewise with scale and shift. Scale is 1 by default and the scale tensor is
    the offset from the default, i.e., if scale == 0 and shift == 0 this method is equivalent to the identity.

    Args:
        x: Input tensor (e.g., input to a transformer block with shape (batch_size, sequence_length, dim)).
        scale: Scale tensor with shape (batch_size, dim) or (batch_size, 1 dim).
        shift: Shift tensor with shape (batch_size, dim) or (batch_size, 1 dim).

    Returns:
        The scaled and shifted input, where each feature of x is scaled by (1 + scale) and then shifted by shift.
    """
    if x.ndim == 3:
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
    return x * (1 + scale) + shift


def modulate_gate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Gates the input x feature-wise with gate.

    Args:
        x: Input tensor (e.g., input to a transformer block with shape (batch_size, sequence_length, dim)).
        gate: Gate tensor with shape (batch_size, dim) or (batch_size, 1 dim).

    Returns:
        The gated input, where each feature of x is multiplied by the corresponding gate value.
    """
    if x.ndim == 3:
        gate = gate.unsqueeze(1)
    return gate * x
