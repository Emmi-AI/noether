#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from enum import Enum

import torch.nn as nn


class Activation(Enum):
    GELU = nn.GELU
    TANH = nn.Tanh
    SIGMOID = nn.Sigmoid
    RELU = nn.ReLU
    LEAKY_RELU = nn.LeakyReLU
    SOFTPLUS = nn.Softplus
    ELU = nn.ELU
    SILU = nn.SiLU

    def build(self) -> nn.Module:
        """Create a new instance of the activation module."""
        return self.value()  # type: ignore[no-any-return]
