#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import abc
from abc import abstractmethod
from collections.abc import Sequence

import torch
import torch.nn as nn

from noether.core.schemas.modules.attention import (
    AttentionConfig,
    AttentionPattern,
    MixedAttentionConfig,
    MultiBranchAnchorAttentionConfig,
    TokenSpec,
)
from noether.modeling.modules.attention.anchor_attention.mixed import MixedAttention


class MissingBranchTokensError(ValueError):
    """Raised when expected tokens for a configured branch are not present."""


class MissingAnchorTokenError(ValueError):
    """Raised when a required anchor token is not present."""


class UnexpectedTokenError(ValueError):
    """Raised when an unexpected token is present."""


class MultiBranchAnchorAttention(nn.Module, metaclass=abc.ABCMeta):
    """A base class for multi-branch anchor-based attention modules with shared parameters between branches.

    Anchor attention limits the self-attention to anchor tokens while other tokens use cross-attention.
    Multiple branches for different modalities use the same linear-projection parameters.
    This base class provides a common constructor, validation logic, and forward method implementation.
    Subclasses only need to implement `_create_attention_patterns` to define their specific attention patterns.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        _config: MultiBranchAnchorAttentionConfig = MultiBranchAnchorAttentionConfig(**config.model_dump())  # type: ignore[no-redef]

        if not _config.branches:
            raise ValueError("The 'branches' list cannot be empty.")

        self.mixed_attention = MixedAttention(
            config=MixedAttentionConfig(
                hidden_dim=_config.hidden_dim,
                num_heads=_config.num_heads,
                use_rope=_config.use_rope,
                bias=_config.bias,
                init_weights=_config.init_weights,
                dropout=_config.dropout,
            )  # type: ignore[call-arg]
        )
        self.branches = _config.branches
        self.anchor_suffix = _config.anchor_suffix

    def forward(
        self,
        x: torch.Tensor,
        token_specs: Sequence[TokenSpec],
        freqs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply attention using the patterns defined by the subclass."""
        self._validate(token_specs)
        patterns = self._create_attention_patterns(token_specs)
        return self.mixed_attention(x, token_specs, patterns, freqs=freqs)  # type: ignore[no-any-return]

    @abstractmethod
    def _create_attention_patterns(self, token_specs: Sequence[TokenSpec]) -> Sequence[AttentionPattern]:
        """Create attention patterns based on the specific attention strategy.

        Args:
            token_specs: Sequence of token specifications defining the input structure.

        Returns:
            List of attention patterns defining the attention behavior.
        """
        raise NotImplementedError("Subclasses must implement _create_attention_patterns.")

    def _validate(self, token_specs: Sequence[TokenSpec]) -> None:
        """Validates that every configured branch and its anchor are present."""
        token_names = {spec.name for spec in token_specs}

        # Validate that no unexpected tokens are present.
        for spec in token_specs:
            if not any(spec.name.startswith(branch) for branch in self.branches):
                raise UnexpectedTokenError(
                    f"Unexpected token in input: Token '{spec.name}' does not belong to any configured branch. "
                    f"Configured branches: {list(self.branches)}"
                )

        # Enforce that every configured branch is present and has an anchor.
        for branch in self.branches:
            if not any(spec.name.startswith(branch) for spec in token_specs):
                raise MissingBranchTokensError(
                    f"Configured branch '{branch}' has no tokens present in the input `token_specs`."
                )

            anchor_name = f"{branch}{self.anchor_suffix}"
            if anchor_name not in token_names:
                raise MissingAnchorTokenError(
                    f"Configured branch '{branch}' is missing its required anchor token '{anchor_name}'."
                )
