#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections import defaultdict
from collections.abc import Sequence

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

    Supports KV caching for efficient inference. When a ``TokenSpec`` has ``size=None``,
    its key/value representations are loaded from the provided ``kv_cache`` instead of
    being computed from the input tensor.

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
        key_padding_mask: torch.Tensor | None = None,
        freqs: torch.Tensor | None = None,
        kv_cache: dict[str, dict[str, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]] | None]:
        """Apply mixed attention with flexible token-name-based patterns.

        Args:
            x: Input tensor [batch_size, n_tokens, dim]. Only contains tokens with
                ``size is not None`` in ``token_specs``.
            token_specs: Sequence of token specifications defining the input structure.
                Tokens with ``size=None`` are loaded from ``kv_cache``.
            attention_patterns: Sequence of attention patterns to apply. Each pattern defines which
                token groups (queries) attend to which other token groups (keys/values).
                The provided patterns must be exhaustive and non-overlapping. This means every
                input (non-cached) token group must be a query in exactly one pattern.
            key_padding_mask: Optional boolean mask of shape ``(batch_size, n_tokens)``.
                ``True`` indicates a real token; ``False`` indicates a padding token that should not be attended to.
                The mask is sliced per attention pattern to cover only the key/value tokens of that pattern.
                Not supported when using KV cache (cached tokens are assumed to be valid).
            freqs: RoPE frequencies for positional encoding (only for input tokens).
            kv_cache: KV cache from a previous forward pass. Structure:
                ``{token_name: {"k": tensor, "v": tensor}}``.

        Returns:
            Tuple of (output, new_kv_cache). Output has the same shape as ``x`` (only input tokens).
            ``new_kv_cache`` contains anchor K/V for future cached inference, or ``None`` when
            using cached tokens.
        """
        # Separate input tokens (in x) from cached tokens (loaded from kv_cache)
        input_specs = [s for s in token_specs if s.size is not None]
        cached_token_names: set[str] = {s.name for s in token_specs if s.size is None}

        self._validate_inputs(x, input_specs, attention_patterns, key_padding_mask, freqs, cached_token_names)

        input_sizes = [spec.size for spec in input_specs]

        def to_token_dict(x: torch.Tensor, split_dim: int = 2) -> dict[str, torch.Tensor]:
            splits = x.split(input_sizes, dim=split_dim)
            return {spec.name: split for spec, split in zip(input_specs, splits, strict=True)}

        if cached_token_names:
            # Cached path: only compute Q (K/V come from cache)
            if not kv_cache:
                raise ValueError("Cannot use cached tokens: kv_cache is empty.")
            q = einops.rearrange(self.q(x), "bs s (nh hd) -> bs nh s hd", nh=self.num_heads)
            if self.use_rope and freqs is not None:
                q = rope(q, freqs=freqs)

            q_dict: dict[str, torch.Tensor] = to_token_dict(q)
            k_dict: dict[str, torch.Tensor] = {}
            v_dict: dict[str, torch.Tensor] = {}

            for name in cached_token_names:
                if name not in kv_cache:
                    raise ValueError(f"Cached token '{name}' not found in kv_cache.")
                k_dict[name] = kv_cache[name]["k"]
                v_dict[name] = kv_cache[name]["v"]

            new_cache = None
        else:
            # Normal path: compute Q, K, V for all input tokens
            q = einops.rearrange(self.q(x), "bs s (nh hd) -> bs nh s hd", nh=self.num_heads)
            k = einops.rearrange(self.k(x), "bs s (nh hd) -> bs nh s hd", nh=self.num_heads)
            v = einops.rearrange(self.v(x), "bs s (nh hd) -> bs nh s hd", nh=self.num_heads)
            if self.use_rope and freqs is not None:
                q, k = rope(q, freqs=freqs), rope(k, freqs=freqs)

            q_dict, k_dict, v_dict = [to_token_dict(x) for x in [q, k, v]]

            # Save anchor K/V for future cached inference
            new_cache = {}
            for spec in input_specs:
                if "_anchor" in spec.name:
                    new_cache[spec.name] = {"k": k_dict[spec.name], "v": v_dict[spec.name]}

        # Build key_padding_mask dict (if provided)
        mask_dict: dict[str, torch.Tensor] | None = None
        if key_padding_mask is not None:
            if cached_token_names:
                raise ValueError("key_padding_mask is not supported when using KV cache.")
            mask_dict = to_token_dict(key_padding_mask, split_dim=1)

        # Filter cached tokens out of pattern query lists (they have no Q, only cached K/V)
        if cached_token_names:
            attention_patterns = [
                AttentionPattern(
                    query_tokens=[name for name in p.query_tokens if name not in cached_token_names],
                    key_value_tokens=p.key_value_tokens,
                )
                for p in attention_patterns
                if any(name not in cached_token_names for name in p.query_tokens)
            ]

        token_outputs = self._process_pattern_batched(
            attention_patterns=attention_patterns,
            q_dict=q_dict,
            k_dict=k_dict,
            v_dict=v_dict,
            mask_dict=mask_dict,
        )

        # Assemble output (only input tokens, in original order)
        output_parts = [token_outputs[spec.name] for spec in input_specs]
        output = torch.cat(output_parts, dim=2)
        output = einops.rearrange(output, "bs nh s hd -> bs s (nh hd)")
        return self.proj(output), new_cache

    def _process_pattern_batched(
        self,
        attention_patterns: Sequence[AttentionPattern],
        q_dict: dict[str, torch.Tensor],
        k_dict: dict[str, torch.Tensor],
        v_dict: dict[str, torch.Tensor],
        mask_dict: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Efficient mixed attention implementation that batches compatible (same shape) attention patterns.

        Args:
            attention_patterns: Attention patterns to process.
            q_dict: Per-token-name query tensors ``(bs, nh, seq, hd)`` (only input tokens).
            k_dict: Per-token-name key tensors ``(bs, nh, seq, hd)`` (input + cached tokens).
            v_dict: Per-token-name value tensors ``(bs, nh, seq, hd)`` (input + cached tokens).
            mask_dict: Per-token-name key padding masks, or None.
        """
        # Group compatible patterns (same query_len, kv_len)
        pattern_groups: dict[tuple[int, int], list[AttentionPattern]] = defaultdict(list)
        for pattern in attention_patterns:
            query_len = sum(q_dict[name].shape[2] for name in pattern.query_tokens)
            kv_len = sum(k_dict[name].shape[2] for name in pattern.key_value_tokens)
            pattern_groups[(query_len, kv_len)].append(pattern)

        token_outputs: dict[str, torch.Tensor] = {}
        for group in pattern_groups.values():
            qs = [torch.cat([q_dict[name] for name in patt.query_tokens], dim=2) for patt in group]
            ks = [torch.cat([k_dict[name] for name in patt.key_value_tokens], dim=2) for patt in group]
            vs = [torch.cat([v_dict[name] for name in patt.key_value_tokens], dim=2) for patt in group]

            # Batch independent attentions on the batch dimension
            q_batched = torch.cat(qs, dim=0)
            k_batched = torch.cat(ks, dim=0)
            v_batched = torch.cat(vs, dim=0)

            attn_mask_batched: torch.Tensor | None = None
            if mask_dict is not None:
                # For each pattern, slice the mask to its KV tokens (mirroring how k/v are assembled).
                # Shape (batch, 1, 1, kv_len): the 1s let PyTorch broadcast the same key mask
                # across all heads and all query positions (every query sees the same valid key set).
                per_pattern_masks = []
                for patt in group:
                    kv_bool = torch.cat([mask_dict[name] for name in patt.key_value_tokens], dim=1)
                    per_pattern_masks.append(kv_bool[:, None, None, :])
                attn_mask_batched = torch.cat(per_pattern_masks, dim=0)

            output_batched = F.scaled_dot_product_attention(
                q_batched,
                k_batched,
                v_batched,
                attn_mask=attn_mask_batched,
                dropout_p=self.dropout if self.training else 0.0,
            )
            # Undo the batch concatenation
            output_chunks = torch.chunk(output_batched, chunks=len(group), dim=0)
            # Undo the sequence concatenation
            for pattern, pattern_output in zip(group, output_chunks, strict=True):
                query_sizes = [q_dict[name].shape[2] for name in pattern.query_tokens]
                for name, chunk in zip(pattern.query_tokens, pattern_output.split(query_sizes, dim=2), strict=True):
                    token_outputs[name] = chunk
        return token_outputs

    def _validate_inputs(
        self,
        x: torch.Tensor,
        input_specs: Sequence[TokenSpec],
        attention_patterns: Sequence[AttentionPattern],
        key_padding_mask: torch.Tensor | None,
        freqs: torch.Tensor | None,
        cached_token_names: set[str] | None = None,
    ) -> None:
        """Validate input consistency."""
        cached_token_names = cached_token_names or set()

        if not self.use_rope == (freqs is not None):
            raise ValueError(f"RoPE usage mismatch: self.use_rope = {self.use_rope}, but freqs is {freqs is not None}")

        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool:
                raise ValueError(f"key_padding_mask must be a bool tensor, got {key_padding_mask.dtype}.")
            if key_padding_mask.ndim != 2:
                raise ValueError(
                    f"key_padding_mask must be 2D with shape (batch_size, n_tokens), got shape {tuple(key_padding_mask.shape)}."
                )
            if key_padding_mask.shape[1] != x.shape[1]:
                raise ValueError(
                    f"key_padding_mask n_tokens dim ({key_padding_mask.shape[1]}) must match x sequence length ({x.shape[1]})."
                )

        # Validate that input specs match the tensor size
        expected_size = sum(spec.size for spec in input_specs if spec.size is not None)
        if expected_size != x.shape[1]:
            raise ValueError(f"Token specs total size {expected_size} != tensor size {x.shape[1]}")

        # Validate attention patterns cover all token specs
        spec_names = {spec.name for spec in input_specs} | cached_token_names
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
