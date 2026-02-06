#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections.abc import Sequence

from noether.core.schemas.modules.attention import (
    AttentionPattern,
    CrossAnchorAttentionConfig,
    TokenSpec,
)
from noether.modeling.modules.attention.anchor_attention.multi_branch import MultiBranchAnchorAttention


class CrossAnchorAttention(MultiBranchAnchorAttention):
    """Anchor attention across branches: each configured branch attends to the anchors of all other branches.

    For a list of branches (e.g., A, B, C), this creates a pattern,
    where A attends to (B_anchors + C_anchors), B attends to (A_anchors + C_anchors), etc.
    It requires all configured branches and their anchors to be present in the input.

    Example: all surface tokens attend to volume_anchors and all volume tokens attend to surface_anchors.

    .. code-block:: python

        AttentionPattern(query_tokens=["surface_anchors", "surface_queries"], key_value_tokens=["volume_anchors"])
        AttentionPattern(query_tokens=["volume_anchors", "volume_queries"], key_value_tokens=["surface_anchors"])
    """

    def __init__(
        self,
        config: CrossAnchorAttentionConfig,
    ):
        if len(config.branches) < 2:
            raise ValueError("CrossAnchorAttention requires at least two branches.")
        super().__init__(
            config=config,
        )

    def _create_attention_patterns(self, token_specs: Sequence[TokenSpec]) -> Sequence[AttentionPattern]:
        """Create cross-attention patterns where each branch attends to other branches' anchors."""
        patterns = []
        for query_branch in self.branches:
            query_tokens = [spec.name for spec in token_specs if spec.name.startswith(query_branch)]
            other_branches = [name for name in self.branches if name != query_branch]
            key_value_anchors = [f"{other_branch}{self.anchor_suffix}" for other_branch in other_branches]
            attention_pattern = AttentionPattern(query_tokens=query_tokens, key_value_tokens=key_value_anchors)
            patterns.append(attention_pattern)
        return patterns
