#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Back-compat re-exports for ``noether.core.schemas.modules``.

Module configs have been moved next to their matching classes in
:mod:`noether.modeling.modules`. Base configs without a matching class
(:class:`AttentionConfig`, :class:`AttentionPattern`, :class:`TokenSpec`stay in :mod:`.attention`.

Concrete attention configs are loaded lazily via :pep:`562` to avoid circular
imports between the schema package and the modeling modules that depend on
:class:`AttentionConfig`.
"""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, Any

# Base configs (no matching class) — eagerly imported from the canonical schema files.
from .attention import (
    AttentionConfig,
    AttentionPattern,
    TokenSpec,
)

# Concrete configs that have moved next to their classes — loaded lazily to break import cycles.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # attention
    "CrossAnchorAttentionConfig": (
        "noether.modeling.modules.attention.anchor_attention.cross",
        "CrossAnchorAttentionConfig",
    ),
    "DotProductAttentionConfig": (
        "noether.modeling.modules.attention.dot_product",
        "DotProductAttentionConfig",
    ),
    "JointAnchorAttentionConfig": (
        "noether.modeling.modules.attention.anchor_attention.joint",
        "JointAnchorAttentionConfig",
    ),
    "MixedAttentionConfig": (
        "noether.modeling.modules.attention.anchor_attention.mixed",
        "MixedAttentionConfig",
    ),
    "MultiBranchAnchorAttentionConfig": (
        "noether.modeling.modules.attention.anchor_attention.multi_branch",
        "MultiBranchAnchorAttentionConfig",
    ),
    "PerceiverAttentionConfig": (
        "noether.modeling.modules.attention.perceiver",
        "PerceiverAttentionConfig",
    ),
    "TransolverAttentionConfig": (
        "noether.modeling.modules.attention.transolver",
        "TransolverAttentionConfig",
    ),
    "TransolverPlusPlusAttentionConfig": (
        "noether.modeling.modules.attention.transolver_plusplus",
        "TransolverPlusPlusAttentionConfig",
    ),
    # blocks
    "PerceiverBlockConfig": ("noether.modeling.modules.blocks.perceiver", "PerceiverBlockConfig"),
    "TransformerBlockConfig": ("noether.modeling.modules.blocks.transformer", "TransformerBlockConfig"),
    # decoders
    "DeepPerceiverDecoderConfig": (
        "noether.modeling.modules.decoders.deep_perceiver",
        "DeepPerceiverDecoderConfig",
    ),
    # encoders
    "SupernodePoolingConfig": (
        "noether.modeling.modules.encoders.supernode_pooling",
        "SupernodePoolingConfig",
    ),
    # layers
    "ContinuousSincosEmbeddingConfig": (
        "noether.modeling.modules.layers.continuous_sincos_embed",
        "ContinuousSincosEmbeddingConfig",
    ),
    "LayerScaleConfig": ("noether.modeling.modules.layers.layer_scale", "LayerScaleConfig"),
    "LinearProjectionConfig": (
        "noether.modeling.modules.layers.linear_projection",
        "LinearProjectionConfig",
    ),
    "RopeFrequencyConfig": ("noether.modeling.modules.layers.rope_frequency", "RopeFrequencyConfig"),
    "UnquantizedDropPathConfig": (
        "noether.modeling.modules.layers.drop_path",
        "UnquantizedDropPathConfig",
    ),
    # mlp
    "MLPConfig": ("noether.modeling.modules.mlp.mlp", "MLPConfig"),
    "UpActDownMLPConfig": ("noether.modeling.modules.mlp.upactdown_mlp", "UpActDownMLPConfig"),
    # untied
    "UntiedLinearConfig": ("noether.modeling.modules.untied", "UntiedLinearConfig"),
    "UntiedMLPConfig": ("noether.modeling.modules.untied", "UntiedMLPConfig"),
    "UntiedMixedAttentionConfig": ("noether.modeling.modules.untied", "UntiedMixedAttentionConfig"),
    "UntiedPerceiverBlockConfig": ("noether.modeling.modules.untied", "UntiedPerceiverBlockConfig"),
    "UntiedTransformerBlockConfig": ("noether.modeling.modules.untied", "UntiedTransformerBlockConfig"),
}

__all__ = [
    "AttentionConfig",
    "AttentionPattern",
    "ContinuousSincosEmbeddingConfig",
    "CrossAnchorAttentionConfig",
    "DeepPerceiverDecoderConfig",
    "DotProductAttentionConfig",
    "JointAnchorAttentionConfig",
    "LayerScaleConfig",
    "LinearProjectionConfig",
    "MLPConfig",
    "MixedAttentionConfig",
    "MultiBranchAnchorAttentionConfig",
    "PerceiverAttentionConfig",
    "PerceiverBlockConfig",
    "RopeFrequencyConfig",
    "SupernodePoolingConfig",
    "TokenSpec",
    "TransformerBlockConfig",
    "TransolverAttentionConfig",
    "TransolverPlusPlusAttentionConfig",
    "UnquantizedDropPathConfig",
    "UntiedLinearConfig",
    "UntiedMLPConfig",
    "UntiedMixedAttentionConfig",
    "UntiedPerceiverBlockConfig",
    "UntiedTransformerBlockConfig",
    "UpActDownMLPConfig",
]


def __getattr__(name: str) -> Any:
    try:
        module_path, attr = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    warnings.warn(
        f"Importing `{name}` from `{__name__}` is deprecated; import from `{module_path}` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(importlib.import_module(module_path), attr)


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))


if TYPE_CHECKING:  # static type checkers — keep in sync with _LAZY_EXPORTS
    from noether.modeling.modules.attention.anchor_attention.cross import CrossAnchorAttentionConfig
    from noether.modeling.modules.attention.anchor_attention.joint import JointAnchorAttentionConfig
    from noether.modeling.modules.attention.anchor_attention.mixed import MixedAttentionConfig
    from noether.modeling.modules.attention.anchor_attention.multi_branch import MultiBranchAnchorAttentionConfig
    from noether.modeling.modules.attention.dot_product import DotProductAttentionConfig
    from noether.modeling.modules.attention.perceiver import PerceiverAttentionConfig
    from noether.modeling.modules.attention.transolver import TransolverAttentionConfig
    from noether.modeling.modules.attention.transolver_plusplus import TransolverPlusPlusAttentionConfig
    from noether.modeling.modules.blocks.perceiver import PerceiverBlockConfig
    from noether.modeling.modules.blocks.transformer import TransformerBlockConfig
    from noether.modeling.modules.decoders.deep_perceiver import DeepPerceiverDecoderConfig
    from noether.modeling.modules.encoders.supernode_pooling import SupernodePoolingConfig
    from noether.modeling.modules.layers.continuous_sincos_embed import ContinuousSincosEmbeddingConfig
    from noether.modeling.modules.layers.drop_path import UnquantizedDropPathConfig
    from noether.modeling.modules.layers.layer_scale import LayerScaleConfig
    from noether.modeling.modules.layers.linear_projection import LinearProjectionConfig
    from noether.modeling.modules.layers.rope_frequency import RopeFrequencyConfig
    from noether.modeling.modules.mlp.mlp import MLPConfig
    from noether.modeling.modules.mlp.upactdown_mlp import UpActDownMLPConfig
    from noether.modeling.modules.untied import (
        UntiedLinearConfig,
        UntiedMixedAttentionConfig,
        UntiedMLPConfig,
        UntiedPerceiverBlockConfig,
        UntiedTransformerBlockConfig,
    )
