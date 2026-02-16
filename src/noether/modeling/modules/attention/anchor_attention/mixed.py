#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections import defaultdict
from collections.abc import Sequence
from itertools import accumulate

import einops
import torch
import torch.nn.functional as F

from noether.core.schemas.modules.attention import AttentionPattern, MixedAttentionConfig, TokenSpec
from noether.modeling.functional.rope import rope
from noether.modeling.modules.attention import DotProductAttention


class MixedAttention(DotProductAttention):
    """Mixed attention with a selectable implementation for performance or readability.

    This module allows for structured attention patterns where different groups of tokens
    (defined by `TokenSpec`) have specific interaction patterns (defined by `AttentionPattern`).
    Instead of full self-attention, you can specify, for example, that one type of
    token can only attend to itself, while another can attend to all tokens.

    This is achieved by splitting the main Q, K, V tensors based on the token specs
    and then performing separate attention computations for each pattern.

    Example input structure (forward pass signature) for implementing Anchor Attention:

    .. code-block:: python

        x = torch.cat([surface_anchors, surface_queries, volume_anchors, volume_queries], dim=1)  # sequence dim
        token_specs = [
            TokenSpec("surface_anchors", 100),
            TokenSpec("surface_queries", 50),
            TokenSpec("volume_anchors", 80),
            TokenSpec("volume_queries", 60),
        ]
        attention_patterns = [
            AttentionPattern(query_tokens=["surface_anchors", "surface_queries"], key_value_tokens=["surface_anchors"]),
            AttentionPattern(query_tokens=["volume_anchors", "volume_queries"], key_value_tokens=["volume_anchors"]),
        ]
    """

    def __init__(
        self,
        config: MixedAttentionConfig,
    ) -> None:
        """
        Args:
            config: Configuration for the MixedAttention module. See
                :class:`~noether.core.schemas.modules.attention.MixedAttentionConfig` for the available options.
        """
        super().__init__(config=config)

    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        token_specs: Sequence[TokenSpec],
        attention_patterns: Sequence[AttentionPattern],
        attention_mask: torch.Tensor | None = None,
        freqs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply mixed attention with flexible token-name-based patterns.

        Args:
            x: Input tensor [batch_size, n_tokens, dim]
            token_specs: Sequence of token specifications defining the input structure: assumes that the input
                x is a concatenation of tokens in the order of token_specs.
            attention_patterns: Sequence of attention patterns to apply. Each pattern defines which
                token groups (queries) attend to which other token groups (keys/values).
                The provided patterns must be exhaustive and non-overlapping. This means every
                token group defined in `token_specs` must be a query in exactly one pattern.
            attention_mask: Optional attention mask (not currently supported)
            freqs: RoPE frequencies for positional encoding
        """
        self._validate_inputs(x, token_specs, attention_patterns, attention_mask, freqs)

        # Initial Projection
        q, k, v = einops.rearrange(
            self.qkv(x), "bs s (three nh hd) -> three bs nh s hd", three=3, nh=self.num_heads
        ).unbind(0)

        if self.use_rope and freqs is not None:
            q, k = rope(q, freqs=freqs), rope(k, freqs=freqs)

        # Prepare token slices and size map helpers for processing the attention patterns
        sizes = [spec.size for spec in token_specs]
        start_indices = [0] + list(accumulate(sizes[:-1]))
        token_slices = {
            s.name: slice(start, s.size + start) for s, start in zip(token_specs, start_indices, strict=False)
        }
        spec_size_map = {spec.name: spec.size for spec in token_specs}

        token_outputs = self._process_pattern_batched(attention_patterns, q, k, v, token_slices, spec_size_map)  # type: ignore[arg-type]

        # Final assembly and output projection
        output_parts = [token_outputs[spec.name] for spec in token_specs]
        output = torch.cat(output_parts, dim=2)
        output = einops.rearrange(output, "bs nh s hd -> bs s (nh hd)")
        return self.proj(output)  # type: ignore[no-any-return]

    def _process_pattern_batched(
        self,
        attention_patterns: Sequence[AttentionPattern],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        token_slices: dict[str, slice],
        spec_size_map: dict[str, int],
    ) -> dict[str, torch.Tensor]:
        """Efficient mixed attention implementation that batches compatible (same shape) attention patterns."""
        # Group compatible patterns
        pattern_groups: dict[tuple[int, int], list[AttentionPattern]] = defaultdict(list)
        for pattern in attention_patterns:
            query_len = sum(spec_size_map[name] for name in pattern.query_tokens)
            kv_len = sum(spec_size_map[name] for name in pattern.key_value_tokens)
            pattern_groups[(query_len, kv_len)].append(pattern)

        token_outputs: dict[str, torch.Tensor] = {}
        for group in pattern_groups.values():
            # Concatenate sequences for each attention pattern (e.g. multiple queries and multiple keys & values)
            qs = [torch.cat([q[:, :, token_slices[name]] for name in patt.query_tokens], dim=2) for patt in group]
            ks = [torch.cat([k[:, :, token_slices[name]] for name in patt.key_value_tokens], dim=2) for patt in group]
            vs = [torch.cat([v[:, :, token_slices[name]] for name in patt.key_value_tokens], dim=2) for patt in group]
            # Concatenate independent attentions on batch dimension to process in parallel (e.g. multiple modalities)
            q_batched = torch.cat(qs, dim=0)
            k_batched = torch.cat(ks, dim=0)
            v_batched = torch.cat(vs, dim=0)
            output_batched = F.scaled_dot_product_attention(
                q_batched, k_batched, v_batched, dropout_p=self.dropout if self.training else 0.0
            )
            # Undo the batch concatenation
            output_chunks = torch.chunk(output_batched, chunks=len(group), dim=0)
            # Undo the sequence concatenation
            for pattern, pattern_output in zip(group, output_chunks, strict=True):
                query_sizes = [spec_size_map[name] for name in pattern.query_tokens]
                for name, chunk in zip(pattern.query_tokens, pattern_output.split(query_sizes, dim=2), strict=True):
                    token_outputs[name] = chunk
        return token_outputs

    def _validate_inputs(
        self,
        x: torch.Tensor,
        token_specs: Sequence[TokenSpec],
        attention_patterns: Sequence[AttentionPattern],
        attention_mask: torch.Tensor | None,
        freqs: torch.Tensor | None,
    ) -> None:
        """Validate input consistency."""
        if not self.use_rope == (freqs is not None):
            raise ValueError(f"RoPE usage mismatch: self.use_rope = {self.use_rope}, but freqs is {freqs is not None}")

        if attention_mask is not None:
            raise NotImplementedError("Attention masks are not supported in this implementation.")

        expected_size = sum(spec.size for spec in token_specs)
        if expected_size != x.shape[1]:
            raise ValueError(f"Token specs total size {expected_size} != tensor size {x.shape[1]}")

        spec_names = {spec.name for spec in token_specs}
        query_names_from_patterns = [name for pattern in attention_patterns for name in pattern.query_tokens]
        if len(query_names_from_patterns) != len(set(query_names_from_patterns)):
            raise ValueError("A token type cannot be a query in multiple attention patterns.")
        if set(query_names_from_patterns) != spec_names:
            raise ValueError("The set of query tokens must exactly match the set of tokens in token_specs.")

        for pattern in attention_patterns:
            for token_name in pattern.key_value_tokens:
                if token_name not in spec_names:
                    raise ValueError(
                        f"Token '{token_name}' in `key_value_tokens` of an attention pattern "
                        "is not defined in `token_specs`."
                    )
