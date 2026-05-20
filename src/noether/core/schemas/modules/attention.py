#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
"""Base attention configs and back-compat re-exports for moved attention configs.

The base configs (:class:`AttentionConfig`, :class:`TokenSpec`,
:class:`AttentionPattern`) have no
matching class and stay here. The concrete attention configs have moved next
to their matching classes in :mod:`noether.modeling.modules.attention`; they
are re-exported here for backward compatibility.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from noether.core.types import InitWeightsMode


# =====================================================================================================================
#                                                   REGULAR ATTENTION
# ---------------------------------------------------------------------------------------------------------------------
class AttentionConfig(BaseModel):
    """
    Configuration for an attention module.
    Since we can have many different attention implementations, we allow extra fields.
    such that we can use the same schema for all attention modules.
    """

    model_config = ConfigDict(extra="allow")

    """Configuration for an attention module."""

    hidden_dim: int = Field(..., ge=1)
    """Dimensionality of the hidden features."""

    num_heads: int = Field(..., ge=1)
    """Number of attention heads."""

    use_rope: bool = Field(False)
    """Whether to use Rotary Positional Embeddings (RoPE)."""

    dropout: float = Field(0.0, ge=0.0, le=1.0)
    """Dropout rate for the attention weights and output projection."""

    init_weights: InitWeightsMode = Field("truncnormal002")
    """Weight initialization strategy."""

    bias: bool = Field(True)
    """Whether to use bias terms in linear layers."""

    head_dim: int | None = Field(None)
    """Dimensionality of each attention head."""

    qk_norm: bool = Field(False)
    """Whether to apply layer normalization to the query and key features before computing attention scores."""

    @model_validator(mode="after")
    def validate_hidden_dim_and_num_heads(self):
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("The 'hidden_dim' must be divisible by 'num_heads'.")
        self.head_dim = self.hidden_dim // self.num_heads
        return self


class TokenSpec(BaseModel):
    """Specification for a token type in the attention mechanism.

    When ``size`` is ``None``, the token group is not present in the input tensor and its
    key/value representations will be loaded from a KV cache instead.
    """

    name: str  # Semantic identifier (e.g., "surface_anchors")
    size: int | None = Field(..., ge=0)  # Number of tokens, or None when loaded from KV cache

    @classmethod
    def from_dict(cls, token_dict: dict[str, int | None]) -> "TokenSpec":
        """Create TokenSpec from dictionary with single key-value pair."""
        if len(token_dict) != 1:
            raise ValueError("Dictionary must contain exactly one key-value pair")
        name, size = next(iter(token_dict.items()))
        return cls(name=name, size=size)

    def to_dict(self) -> dict[str, int | None]:
        """Convert TokenSpec to dictionary."""
        return {self.name: self.size}

    @computed_field  # type: ignore[misc]
    @property
    def domain(self) -> str:
        """Extract token domain from the name (e.g., "surface" from "surface_anchors")."""
        return self.name.split("_")[0]

    @computed_field  # type: ignore[misc]
    @property
    def attn_type(self) -> str:
        """Extract attention type from the name (e.g., "anchors" from "surface_anchors")."""
        return self.name.split("_")[-1]


class AttentionPattern(BaseModel):
    """Defines which tokens attend to which other tokens."""

    query_tokens: Sequence[str]  #  The tokens that attend to the key/value tokens, e.g. ["anchors", "queries"]
    key_value_tokens: Sequence[str]  # The tokens that are attended to by the query tokens, e.g. ["anchors"]


# =====================================================================================================================
# Lazy back-compat re-exports for configs that have moved next to their matching classes.
# Lazy loading is required to avoid circular imports: the new homes import ``AttentionConfig``
# from this module, so eager re-imports would cycle when this module is loaded as part of
# loading those classes.
# ---------------------------------------------------------------------------------------------------------------------
import importlib  # noqa: E402
import warnings  # noqa: E402

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DotProductAttentionConfig": ("noether.modeling.modules.attention.dot_product", "DotProductAttentionConfig"),
    "PerceiverAttentionConfig": ("noether.modeling.modules.attention.perceiver", "PerceiverAttentionConfig"),
    "TransolverAttentionConfig": ("noether.modeling.modules.attention.transolver", "TransolverAttentionConfig"),
    "TransolverPlusPlusAttentionConfig": (
        "noether.modeling.modules.attention.transolver_plusplus",
        "TransolverPlusPlusAttentionConfig",
    ),
    "MixedAttentionConfig": ("noether.modeling.modules.attention.anchor_attention.mixed", "MixedAttentionConfig"),
    "MultiBranchAnchorAttentionConfig": (
        "noether.modeling.modules.attention.anchor_attention.multi_branch",
        "MultiBranchAnchorAttentionConfig",
    ),
    "CrossAnchorAttentionConfig": (
        "noether.modeling.modules.attention.anchor_attention.cross",
        "CrossAnchorAttentionConfig",
    ),
    "JointAnchorAttentionConfig": (
        "noether.modeling.modules.attention.anchor_attention.joint",
        "JointAnchorAttentionConfig",
    ),
}

__all__ = [
    "AttentionConfig",
    "AttentionPattern",
    "CrossAnchorAttentionConfig",
    "DotProductAttentionConfig",
    "JointAnchorAttentionConfig",
    "MixedAttentionConfig",
    "MultiBranchAnchorAttentionConfig",
    "PerceiverAttentionConfig",
    "TokenSpec",
    "TransolverAttentionConfig",
    "TransolverPlusPlusAttentionConfig",
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
