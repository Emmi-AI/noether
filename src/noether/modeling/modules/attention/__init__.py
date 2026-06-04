#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from torch import nn

from .dot_product import DotProductAttention, DotProductAttentionConfig
from .perceiver import PerceiverAttention, PerceiverAttentionConfig
from .transolver import TransolverAttention, TransolverAttentionConfig
from .transolver_plusplus import TransolverPlusPlusAttention, TransolverPlusPlusAttentionConfig

ATTENTION_REGISTRY: dict[str, type[nn.Module]] = {
    "dot_product": DotProductAttention,
    "perceiver": PerceiverAttention,
    "transolver": TransolverAttention,
    "transolver_plusplus": TransolverPlusPlusAttention,
}

ATTN_IMPL_REGISTRY: set[str] = {
    "sdpa",
    "flash_attn", # flash_attention is only supported by kernels for now
}

__all__ = [
    "DotProductAttention",
    "PerceiverAttention",
    "TransolverAttention",
    "TransolverPlusPlusAttention",
    "DotProductAttentionConfig",
    "PerceiverAttentionConfig",
    "TransolverAttentionConfig",
    "TransolverPlusPlusAttentionConfig",
    "ATTENTION_REGISTRY",
    "ATTN_IMPL_REGISTRY",
]
