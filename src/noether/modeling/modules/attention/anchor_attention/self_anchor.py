#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections.abc import Sequence

from noether.core.schemas.modules.attention import AttentionPattern, TokenSpec
from noether.modeling.modules.attention.anchor_attention.multi_branch import MultiBranchAnchorAttention


class SelfAnchorAttention(MultiBranchAnchorAttention):
    """Anchor attention within branches: each configured branch attends to its own anchors independently.

    For a list of branches (e.g., A, B, C), this creates a pattern where A tokens attend to A_anchors,
    B tokens attend to B_anchors, and C tokens attend to C_anchors.
    It requires all configured branches and their anchors to be present in the input.

    Example: surface tokens attend to surface_anchors and volume tokens attend to volume_anchors.

    This is achieved via the following attention patterns:

    .. code-block:: python

        AttentionPattern(query_tokens=["surface_anchors", "surface_queries"], key_value_tokens=["surface_anchors"])
        AttentionPattern(query_tokens=["volume_anchors", "volume_queries"], key_value_tokens=["volume_anchors"])
    """

    def _create_attention_patterns(self, token_specs: Sequence[TokenSpec]) -> Sequence[AttentionPattern]:
        """Create self-attention patterns where each branch attends to its own anchors."""
        patterns = []
        for branch in self.branches:
            anchor_name = f"{branch}{self.anchor_suffix}"
            branch_query_tokens = [spec.name for spec in token_specs if spec.name.startswith(branch)]
            attention_pattern = AttentionPattern(query_tokens=branch_query_tokens, key_value_tokens=[anchor_name])
            patterns.append(attention_pattern)
        return patterns
