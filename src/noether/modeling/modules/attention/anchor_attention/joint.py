#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections.abc import Sequence

from noether.core.schemas.modules.attention import (
    AttentionPattern,
    JointAnchorAttentionConfig,
    TokenSpec,
)
from noether.modeling.modules.attention.anchor_attention.multi_branch import MultiBranchAnchorAttention


class JointAnchorAttention(MultiBranchAnchorAttention):
    """Anchor attention within and across branches: all tokens attend to anchors from all configured branches.

    For a list of branches (e.g., A, B, C), this creates a pattern where all tokens
    (A_anchors, A_queries, B_anchors, B_queries, C_anchors, C_queries) attend to (A_anchors + B_anchors + C_anchors).
    It requires at least one anchor token to be present in the input.

    Example: all tokens attend to (surface_anchors, volume_anchors).
    This is achieved via the following attention pattern:

    .. code-block:: python

        AttentionPattern(
            query_tokens=["surface_anchors", "surface_queries", "volume_anchors", "volume_queries"],
            key_value_tokens=["surface_anchors", "volume_anchors"],
        )

    """

    def __init__(
        self,
        config: JointAnchorAttentionConfig,
    ):
        if len(config.branches) < 2:
            raise ValueError("JointAnchorAttention requires at least two branches. Otherwise use SelfAnchorAttention.")
        super().__init__(
            config=config,
        )

    def _create_attention_patterns(self, token_specs: Sequence[TokenSpec]) -> Sequence[AttentionPattern]:
        """Create joint attention pattern where all tokens attend to all anchors."""
        all_anchor_tokens = [f"{branch}{self.anchor_suffix}" for branch in self.branches]
        all_query_tokens = [spec.name for spec in token_specs]
        attention_pattern = AttentionPattern(query_tokens=all_query_tokens, key_value_tokens=all_anchor_tokens)
        return [attention_pattern]
