#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from torch import nn

from .dot_product import DotProductAttention
from .flare import FLAREAttention
from .perceiver import PerceiverAttention
from .transolver import TransolverAttention
from .transolver_plusplus import TransolverPlusPlusAttention

ATTENTION_REGISTRY: dict[str, type[nn.Module]] = {
    "dot_product": DotProductAttention,
    "flare": FLAREAttention,
    "perceiver": PerceiverAttention,
    "transolver": TransolverAttention,
    "transolver_plusplus": TransolverPlusPlusAttention,
}

__all__ = [
    "DotProductAttention",
    "FLAREAttention",
    "PerceiverAttention",
    "TransolverAttention",
    "TransolverPlusPlusAttention",
    "ATTENTION_REGISTRY",
]
