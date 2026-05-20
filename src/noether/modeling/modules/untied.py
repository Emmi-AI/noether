#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import math
from collections.abc import Sequence
from typing import Any, cast

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field, computed_field

from noether.core.schemas.modules.attention import AttentionConfig, AttentionPattern, TokenSpec
from noether.modeling.functional.modulation import modulate_gate, modulate_scale_shift
from noether.modeling.functional.rope import rope
from noether.modeling.modules.activations import Activation
from noether.modeling.modules.attention.anchor_attention.mixed import MixedAttention, MixedAttentionConfig
from noether.modeling.modules.attention.anchor_attention.multi_branch import MultiBranchAnchorAttention
from noether.modeling.modules.attention.perceiver import PerceiverAttention
from noether.modeling.modules.blocks.perceiver import PerceiverBlock, PerceiverBlockConfig
from noether.modeling.modules.blocks.transformer import TransformerBlock, TransformerBlockConfig
from noether.modeling.modules.layers.linear_projection import LinearProjectionConfig
from noether.modeling.modules.mlp.mlp import MLPConfig


class UntiedLinearConfig(BaseModel):
    """Configuration for a linear layer with per-type (untied) weight banks.

    Composes a :class:`LinearProjectionConfig` (shared across types) with a
    ``num_types`` field: each token type gets its own independent weight matrix
    with the geometry described by the linear projection config.
    """

    num_types: int = Field(..., ge=1)
    """Number of distinct token types, each with its own weight bank."""

    linear_projection: LinearProjectionConfig
    """Shared geometry (input/output dims, bias, init) for every per-type weight bank."""


class UntiedMixedAttentionConfig(MixedAttentionConfig):
    """Configuration for multi-head attention with per-type (untied) QKV and output projections.

    Extends :class:`MixedAttentionConfig` with a ``num_types`` field: the QKV and
    output projections are :class:`UntiedLinear` layers so each token type gets
    its own projection weights. Attention itself is still computed across all
    tokens via :meth:`MixedAttention._process_pattern_batched`.
    """

    num_types: int = Field(..., ge=1)
    """Number of distinct token types, each with its own QKV/output weight bank."""

    @computed_field
    def projection_config(self) -> UntiedLinearConfig:
        """Configuration for the per-type QKV and output projections."""
        return UntiedLinearConfig(
            num_types=self.num_types,
            linear_projection=LinearProjectionConfig(
                input_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                bias=self.bias,
                init_weights=self.init_weights,
            ),
        )


class UntiedMLPConfig(BaseModel):
    """Configuration for an MLP with per-type (untied) weights.

    Composes an :class:`MLPConfig` (architecture: dims, activation, init) with a
    ``num_types`` field. The untied MLP mirrors :class:`MLP`'s topology
    (``input -> [hidden]*(num_layers+1) -> output`` with activations between layers)
    but uses :class:`UntiedLinear` for every linear layer.
    """

    num_types: int = Field(..., ge=1)
    """Number of distinct token types."""

    mlp: MLPConfig
    """Underlying MLP architecture (dims, activation, init)."""


class UntiedTransformerBlockConfig(BaseModel):
    """Configuration for a transformer block with per-type (untied) attention and MLP weights.

    Composes a :class:`TransformerBlockConfig` (shared layout: dims, heads, layer
    scale, drop path, etc.) with a ``num_types`` field. Both sub-layers have
    per-type weights: :class:`UntiedMultiHeadAttention` for attention and
    :class:`UntiedMLP` for the feed-forward.
    """

    num_types: int = Field(..., ge=1)
    """Number of distinct token types for the untied MLP."""

    transformer_block: TransformerBlockConfig
    """Shared transformer-block layout (dims, heads, layer scale, drop path, etc.)."""

    @computed_field
    def attention_config(self) -> UntiedMixedAttentionConfig:
        """Configuration for the UntiedMultiHeadAttention sub-layer."""
        block = self.transformer_block
        return UntiedMixedAttentionConfig(
            **block.model_dump(), **(block.attention_arguments or {}), num_types=self.num_types
        )

    @computed_field
    def untied_mlp_config(self) -> UntiedMLPConfig:
        """Configuration for the UntiedMLP sub-layer."""
        block = self.transformer_block
        # `mlp_hidden_dim` is guaranteed non-None by TransformerBlockConfig's validator.
        assert block.mlp_hidden_dim is not None
        return UntiedMLPConfig(
            num_types=self.num_types,
            mlp=MLPConfig(
                input_dim=block.hidden_dim,
                output_dim=block.hidden_dim,
                hidden_dim=block.mlp_hidden_dim,
                num_layers=0,
                activation="GELU",
                init_weights=block.init_weights,
            ),
        )


class UntiedPerceiverBlockConfig(BaseModel):
    """Configuration for a perceiver block with per-type (untied) Q/output projections and MLP weights.

    Composes a :class:`PerceiverBlockConfig` (shared layout: dims, heads, layer
    scale, drop path, etc.) with a ``num_types`` field. The Q and output
    projections in :class:`PerceiverAttention` become per-type via
    :class:`UntiedLinear`, while the KV projection stays shared (it operates on
    a single geometry encoding). The MLP is also replaced with
    :class:`UntiedMLP`.
    """

    num_types: int = Field(..., ge=1)
    """Number of distinct token types for the untied projections."""

    perceiver_block_config: PerceiverBlockConfig
    """Shared perceiver-block layout (dims, heads, kv_dim, layer scale, drop path, etc.)."""

    @computed_field
    def untied_mlp_config(self) -> UntiedMLPConfig:
        """Configuration for the UntiedMLP sub-layer."""
        block = self.perceiver_block_config
        assert block.mlp_hidden_dim is not None
        return UntiedMLPConfig(
            num_types=self.num_types,
            mlp=MLPConfig(
                input_dim=block.hidden_dim,
                output_dim=block.hidden_dim,
                hidden_dim=block.mlp_hidden_dim,
                num_layers=0,
                activation="GELU",
                init_weights=block.init_weights,
            ),
        )


def _domain_group_sizes(token_specs: Sequence[TokenSpec]) -> list[int]:
    """Sum token sizes per domain, verifying that domains are strictly consecutive.

    Domains are derived from :attr:`TokenSpec.domain` (first ``_``-delimited segment
    of the name, e.g. ``"surface"`` from ``"surface_anchors"``).  Multiple specs
    can belong to the same domain (e.g. ``surface_anchors`` and ``surface_queries``);
    their sizes are summed into a single group.

    Raises:
        ValueError: If specs of the same domain are not adjacent (e.g.
            ``[surface_anchors, volume_anchors, surface_queries]`` has
            ``"surface"`` split across non-adjacent positions).

    Returns:
        Per-domain sizes in the order the domains first appear.
    """
    sizes: list[int] = []
    seen_domains: set[str] = set()
    current_domain: str | None = None
    current_size = 0

    for spec in token_specs:
        if spec.size is None or spec.size == 0:
            continue
        domain = spec.domain
        if domain != current_domain:
            if domain in seen_domains:
                raise ValueError(
                    f"Domain '{domain}' is not consecutive in token_specs. Specs for the same domain must be adjacent."
                )
            if current_domain is not None:
                sizes.append(current_size)
                seen_domains.add(current_domain)
            current_domain = domain
            current_size = spec.size
        else:
            current_size += spec.size

    if current_domain is not None:
        sizes.append(current_size)

    return sizes


class UntiedLinear(nn.Module):
    """Linear layer with per-domain weight banks.

    Groups token specs by :attr:`TokenSpec.domain` and applies a separate
    ``F.linear`` per domain. Automatically picks the fastest strategy at runtime:

    1. ``torch._grouped_mm`` (CUDA, equal-size groups)
    2. Reshape + single ``torch.bmm`` (equal-size groups, any device)
    3. Padded ``torch.bmm`` (moderate skew)
    4. Split + ``F.linear`` loop (heavy skew or very few groups)

    Per-type weights are stored as independent 2D :class:`nn.Parameter` entries
    in an :class:`nn.ParameterList` (one matrix per type). The bmm-based fast
    paths stack them into a 3D tensor on the fly. Storing them as 2D is what
    lets :class:`torch.optim.Muon` (which rejects non-2D parameters) update each
    type's weight matrix independently.

    Domains must be strictly consecutive in ``token_specs``.

    Args:
        config: Number of types and shared linear-projection geometry.
    """

    bias: nn.ParameterList | None

    def __init__(self, config: UntiedLinearConfig) -> None:
        super().__init__()
        self.num_types = config.num_types
        proj = config.linear_projection
        self.init_weights = proj.init_weights
        self.input_dim = proj.input_dim
        self.output_dim = proj.output_dim
        self.weight = nn.ParameterList(
            [nn.Parameter(torch.empty(proj.output_dim, proj.input_dim)) for _ in range(config.num_types)]
        )
        if proj.bias:
            self.bias = nn.ParameterList([nn.Parameter(torch.zeros(proj.output_dim)) for _ in range(config.num_types)])
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the per-type weight banks following :class:`LinearProjection`'s scheme.

        Supported modes (from ``config.linear_projection.init_weights``):
        ``"torch"`` (PyTorch ``nn.Linear`` defaults, applied per type),
        ``"truncnormal"`` / ``"truncnormal002"`` (truncated normal, std=0.02, zero bias),
        ``"zeros"`` (zero weight and bias).
        """
        if self.init_weights == "torch":
            for i in range(self.num_types):
                nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
                if self.bias is not None:
                    bound = 1 / math.sqrt(self.input_dim)
                    nn.init.uniform_(self.bias[i], -bound, bound)
        elif self.init_weights in ("truncnormal", "truncnormal002"):
            for i in range(self.num_types):
                nn.init.trunc_normal_(self.weight[i], std=0.02)
            if self.bias is not None:
                for i in range(self.num_types):
                    nn.init.zeros_(self.bias[i])
        elif self.init_weights == "zeros":
            for i in range(self.num_types):
                nn.init.zeros_(self.weight[i])
            if self.bias is not None:
                for i in range(self.num_types):
                    nn.init.zeros_(self.bias[i])
        else:
            raise NotImplementedError(f"Initialization method {self.init_weights!r} not implemented for UntiedLinear.")

    def _stacked_weight(self) -> torch.Tensor:
        """Per-type weights stacked into a single ``(num_types, output_dim, input_dim)`` tensor."""
        return torch.stack(list(self.weight))

    def _stacked_bias(self) -> torch.Tensor | None:
        """Per-type biases stacked into ``(num_types, output_dim)``, or ``None``."""
        if self.bias is None:
            return None
        return torch.stack(list(self.bias))

    # ------------------------------------------------------------------
    # Forward — auto-selects the fastest code path
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, token_specs: Sequence[TokenSpec]) -> torch.Tensor:
        """Apply per-domain linear projections.

        Args:
            x: Input tensor ``(B, S, D_in)`` with tokens concatenated in
                ``token_specs`` order.  Specs for the same domain must be adjacent.
            token_specs: Token specifications whose sizes sum to ``S``.

        Returns:
            Output tensor ``(B, S, D_out)`` in the same positional order as ``x``.
        """
        sizes = _domain_group_sizes(token_specs)
        B, S, D_in = x.shape
        D_out = self.output_dim
        T = len(sizes)

        if T == 0:
            return x.new_empty(B, 0, D_out)

        # 1) Grouped GEMM via F.grouped_mm (when available).
        # currently only optmized for bfloat16 on CUDA.
        if x.dtype == torch.bfloat16:
            try:
                return self._grouped_mm_forward(x, sizes, B, S, D_in, D_out, T)
            except (RuntimeError, NotImplementedError, ValueError):
                pass

        # 2) Equal-size reshape + bmm (fastest for balanced distributions).
        if all(s == sizes[0] for s in sizes):
            return self._equal_size_bmm_forward(x, sizes[0], B, S, D_in, D_out, T)

        # 3) Padded bmm when overhead is modest (T * max_n ≤ 1.1 * S).
        max_n = max(sizes)
        if max_n * T <= 1.1 * S:
            return self._padded_bmm_forward(x, sizes, B, S, D_in, D_out, T, max_n)

        # 4) Split + F.linear loop (fallback for heavy skew).
        return self._split_loop_forward(x, sizes, D_out)

    def _split_loop_forward(self, x: torch.Tensor, sizes: list[int], D_out: int) -> torch.Tensor:
        """Split along seq-dim and apply :func:`F.linear` per type."""
        chunks = x.split(sizes, dim=1)
        out_chunks: list[torch.Tensor] = []
        for t, chunk in enumerate(chunks):
            if chunk.shape[1] > 0:
                bias_t = self.bias[t] if self.bias is not None else None
                out_chunks.append(F.linear(chunk, self.weight[t], bias_t))
            else:
                out_chunks.append(chunk.new_empty(x.shape[0], 0, D_out))
        return torch.cat(out_chunks, dim=1)

    def _weight_for_bmm(self) -> torch.Tensor:
        """Return weight shaped ``(T, D_in, D_out)`` for right-multiplication.

        Skips the transpose when ``D_in == D_out`` (square weight matrices).
        """
        w = self._stacked_weight()
        if w.shape[-2] == w.shape[-1]:
            return w
        return w.transpose(-2, -1).contiguous()

    def _equal_size_bmm_forward(
        self, x: torch.Tensor, n: int, B: int, S: int, D_in: int, D_out: int, T: int
    ) -> torch.Tensor:
        """Reshape + bmm when every domain has ``n`` tokens."""
        x_r = x.view(B, T, n, D_in).transpose(0, 1).contiguous()
        out = torch.bmm(x_r.view(T, B * n, D_in), self._weight_for_bmm())
        out = out.view(T, B, n, D_out).transpose(0, 1).contiguous().view(B, S, D_out)
        bias = self._stacked_bias()
        if bias is not None:
            out = out + bias.repeat_interleave(n, dim=0).view(1, S, D_out)
        return out

    def _padded_bmm_forward(
        self,
        x: torch.Tensor,
        sizes: list[int],
        B: int,
        S: int,
        D_in: int,
        D_out: int,
        T: int,
        max_n: int,
    ) -> torch.Tensor:
        """Pad each chunk to ``max_n``, one bmm, then unpad."""
        chunks = x.split(sizes, dim=1)
        padded = torch.stack([F.pad(c, (0, 0, 0, max_n - c.shape[1])) for c in chunks])
        out = torch.bmm(padded.view(T, B * max_n, D_in), self._weight_for_bmm())
        out = out.view(T, B, max_n, D_out)
        result = torch.cat([out[t, :, : sizes[t]] for t in range(T)], dim=1)
        bias = self._stacked_bias()
        if bias is not None:
            type_ids = torch.cat([torch.full((s,), t, dtype=torch.long, device=x.device) for t, s in enumerate(sizes)])
            result = result + bias[type_ids].view(1, S, D_out)
        return result

    def _grouped_mm_forward(
        self, x: torch.Tensor, sizes: list[int], B: int, S: int, D_in: int, D_out: int, T: int
    ) -> torch.Tensor:
        """Apply per-domain projections via :func:`torch.nn.functional.grouped_mm`.

        Uses the 3D path when all domain sizes are equal (no ``offs`` needed),
        or the 2D+offs path for unequal sizes.

        Raises:
            RuntimeError: When ``grouped_mm`` is not supported (wrong device,
                unaligned strides, unsupported dtype, …).
            NotImplementedError: When ``grouped_mm`` is not available.
            ValueError: When ``B > 1`` with unequal domain sizes (the 2D+offs
                path only supports ``B=1``).
        """
        w = self._weight_for_bmm()
        bias = self._stacked_bias()
        if all(s == sizes[0] for s in sizes):
            # Equal-size: use 3D mat_a → no offs, cleanest path.
            n = sizes[0]
            # (B, S, D_in) → (B, T, n, D_in) → (T, B, n, D_in) → (T, B*n, D_in)
            mat_a = x.view(B, T, n, D_in).transpose(0, 1).contiguous().view(T, B * n, D_in)
            out = F.grouped_mm(mat_a, w, bias=bias)  # (T, B*n, D_out)
            return out.view(T, B, n, D_out).transpose(0, 1).contiguous().view(B, S, D_out)
        else:
            # Unequal-size: 2D mat_a + offs. Requires per-batch flattening
            # where all of type-t's tokens are contiguous — achievable by
            # permuting (B, [n0|n1|...], D) to type-major order within each batch.
            # For B>1 this would repeat each group's rows B times, so we only
            # take this path for B=1 to keep it simple.
            if B != 1:
                raise ValueError("grouped_mm with unequal domain sizes requires B=1")
            mat_a = x.view(S, D_in)
            offs = torch.tensor(sizes, dtype=torch.int32, device=x.device).cumsum(0, dtype=torch.int32)
            out = F.grouped_mm(mat_a, w, offs=offs, bias=bias)  # (S, D_out)
            return out.view(1, S, D_out)


class UntiedMixedAttention(MixedAttention):
    """Multi-head attention with per-type QKV and output projections.

    Each token type has its own QKV and output projection weights (via
    :class:`UntiedLinear`). The attention computation is pattern-aware, mirroring
    :class:`~noether.modeling.modules.attention.anchor_attention.mixed.MixedAttention`:

    - If ``attention_patterns`` is supplied (typically by a wrapping
      :class:`~noether.modeling.modules.attention.anchor_attention.multi_branch.MultiBranchAnchorAttention`
      such as ``SelfAnchorAttention``/``CrossAnchorAttention``/``JointAnchorAttention``),
      those patterns are honored.
    - If no patterns are supplied (standalone usage), it falls back to a single
      all-to-all pattern — every token attends to every other token.

    This lets the same module act as either a drop-in pattern-free attention or
    the inner workhorse of a multi-branch anchor attention, while keeping QKV
    and output projection weights untied per token type.

    Args:
        config: Configuration specifying attention dims and ``num_types``.
    """

    def __init__(self, config: UntiedMixedAttentionConfig) -> None:
        super().__init__(config=config)

        # Replace the shared nn.Linear projections with per-type untied ones.
        self.q = UntiedLinear(config.projection_config)  # type: ignore[arg-type,assignment]
        self.k = UntiedLinear(config.projection_config)  # type: ignore[arg-type,assignment]
        self.v = UntiedLinear(config.projection_config)  # type: ignore[arg-type,assignment]
        self.proj = UntiedLinear(config.projection_config)  # type: ignore[arg-type,assignment]
        if config.qk_norm:
            self.q_norm: nn.Module = nn.RMSNorm(self.head_dim)
            self.k_norm: nn.Module = nn.RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        token_specs: Sequence[TokenSpec],
        attention_patterns: Sequence[AttentionPattern],
        key_padding_mask: torch.Tensor | None = None,
        freqs: torch.Tensor | None = None,
        kv_cache: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Apply attention with per-type QKV/output projections.

        Positional argument order matches :meth:`MixedAttention.forward` so this
        module is a drop-in replacement for the shared ``MixedAttention`` that
        :class:`MultiBranchAnchorAttention` owns — a wrapping anchor attention
        can call ``self.mixed_attention(x, token_specs, patterns, ...)`` unchanged.

        Args:
            x: Input tensor ``(B, S, D)`` with tokens concatenated in ``token_specs`` order.
                ``S`` must equal the sum of ``token_specs`` sizes.
            token_specs: Token specifications defining the input structure.
            attention_patterns: Optional attention patterns. When ``None``,
                falls back to a single all-to-all pattern over every name in
                ``token_specs`` (standalone usage).
            key_padding_mask: Optional boolean mask ``(B, S)``. ``True`` = real token.
            freqs: RoPE frequencies for positional encoding.
            kv_cache: Optional dictionary for caching key/value tensors. Not yet supported.

        Returns:
            Output tensor ``(B, S, D)``.
        """
        if kv_cache is not None:
            raise NotImplementedError("UntiedMixedAttention does not yet support kv caching.")

        input_specs = [s for s in token_specs if s.size is not None]
        self._validate_inputs(x, input_specs, attention_patterns, key_padding_mask, freqs)

        q = self.q_norm(einops.rearrange(self.q(x, token_specs), "bs s (nh hd) -> bs nh s hd", nh=self.num_heads))
        k = self.k_norm(einops.rearrange(self.k(x, token_specs), "bs s (nh hd) -> bs nh s hd", nh=self.num_heads))
        v = einops.rearrange(self.v(x, token_specs), "bs s (nh hd) -> bs nh s hd", nh=self.num_heads)

        output, _, _ = self._attend(
            q,
            k,
            v,
            input_specs=input_specs,
            attention_patterns=attention_patterns,
            key_padding_mask=key_padding_mask,
            freqs=freqs,
        )

        # Per-type output projection.
        projected: torch.Tensor = self.proj(output, token_specs)
        return projected


class UntiedPerceiverAttention(PerceiverAttention):
    """Perceiver cross-attention with per-type Q and output projections.

    The Q and output projections are :class:`UntiedLinear` layers (one weight
    bank per token type), while the KV projection remains a shared
    ``nn.Linear`` since it operates on a single source (e.g. geometry encoding).

    Args:
        config: Attention configuration (``AttentionConfig``-compatible).
        num_types: Number of distinct token types for per-type projections.
    """

    def __init__(self, config: AttentionConfig, num_types: int) -> None:
        super().__init__(config=config)
        # Replace the shared Q and output projections with per-type versions.
        hidden_dim = self.num_heads * self.head_dim
        self.q = UntiedLinear(  # type: ignore[assignment]
            UntiedLinearConfig(
                num_types=num_types,
                linear_projection=LinearProjectionConfig(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    bias=self.q.bias is not None,
                    init_weights=self.init_weights,
                ),
            )
        )
        self.proj = UntiedLinear(  # type: ignore[assignment]
            UntiedLinearConfig(
                num_types=num_types,
                linear_projection=LinearProjectionConfig(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    bias=self.proj.bias is not None,
                    init_weights=self.init_weights,
                ),
            )
        )

    def forward(  # type: ignore[override]
        self,
        q: torch.Tensor,
        token_specs: Sequence[TokenSpec],
        kv: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        q_freqs: torch.Tensor | None = None,
        k_freqs: torch.Tensor | None = None,
        kv_cache: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Forward pass with per-type Q and output projections.

        Args:
            q: Query tensor ``(B, S, hidden_dim)`` with tokens concatenated
                in ``token_specs`` order.
            token_specs: Token specifications defining the per-type structure.
            kv: Key/value tensor from a single source (e.g. geometry).
                Can be ``None`` when ``kv_cache`` is provided.
            attn_mask: Optional attention mask.
            q_freqs: RoPE frequencies for queries.
            k_freqs: RoPE frequencies for keys.
            kv_cache: Cached K/V tensors from a previous forward pass.

        Returns:
            Tuple of (output, new_kv_cache).
        """
        # Per-type Q projection.
        q = self.q(q, token_specs)
        q = self.q_norm(
            einops.rearrange(
                q,
                "bs seqlen_q (num_heads head_dim) -> bs num_heads seqlen_q head_dim",
                num_heads=self.num_heads,
                head_dim=self.head_dim,
            )
        )

        if kv_cache is not None:
            k = kv_cache["k"]
            v = kv_cache["v"]
            new_cache = None
        else:
            if kv is None:
                raise ValueError("Either kv or kv_cache must be provided.")
            kv_weight = torch.cat([self.k.weight, self.v.weight], dim=0)
            kv_bias = torch.cat([self.k.bias, self.v.bias], dim=0) if self.k.bias is not None else None
            kv_proj = F.linear(kv, kv_weight, kv_bias)

            k, v = einops.rearrange(
                kv_proj,
                "bs seqlen_kv (two num_heads head_dim) -> two bs num_heads seqlen_kv head_dim",
                two=2,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
            ).unbind(0)
            k = self.k_norm(k)

            if self.use_rope:
                assert k_freqs is not None
                k = rope(k, freqs=k_freqs)

            new_cache = {"k": k, "v": v}

        if self.use_rope:
            assert q_freqs is not None
            q = rope(q, freqs=q_freqs)

        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")

        # Per-type output projection.
        return self.proj_dropout(self.proj(x, token_specs)), new_cache


class UntiedMLP(nn.Module):
    """Multi-layer perceptron with per-type weights.

    Mirrors the topology of :class:`MLP` (input -> [hidden]*(num_layers+1) -> output,
    with activations between layers), but uses :class:`UntiedLinear` for every
    linear layer so each token type learns independent weights.

    Args:
        config: Configuration for the untied MLP.
    """

    def __init__(self, config: UntiedMLPConfig) -> None:
        super().__init__()
        mlp = config.mlp
        # Layer dims mirror MLP: input -> hidden (x num_layers+1) -> output.
        dims = [mlp.input_dim] + [mlp.hidden_dim] * (mlp.num_layers + 1) + [mlp.output_dim]
        self.layers = nn.ModuleList(
            [
                UntiedLinear(
                    UntiedLinearConfig(
                        num_types=config.num_types,
                        linear_projection=LinearProjectionConfig(
                            input_dim=in_d,
                            output_dim=out_d,
                            init_weights=mlp.init_weights,
                        ),
                    )
                )
                for in_d, out_d in zip(dims[:-1], dims[1:], strict=True)
            ]
        )
        self.activation = Activation[mlp.activation].build()

    def forward(self, x: torch.Tensor, token_specs: Sequence[TokenSpec]) -> torch.Tensor:
        """Apply the untied MLP.

        Args:
            x: Input tensor ``(B, S, mlp_config.input_dim)``.
            token_specs: Token specifications defining the input structure.

        Returns:
            Output tensor ``(B, S, mlp_config.output_dim)``.
        """
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            x = layer(x, token_specs)
            if i < last:
                x = self.activation(x)
        return x


class UntiedTransformerBlock(TransformerBlock):
    """Pre-norm transformer block with per-type (untied) attention and MLP weights.

    Same architecture and control flow as :class:`TransformerBlock`; only the
    attention and MLP sub-modules differ. The parent's ``__init__`` builds all
    the shared plumbing (norms, modulation, layer scale, drop path, and the
    full forward pass) — including whichever attention module the configured
    ``attention_constructor`` selects. This subclass then:

    1. Replaces ``self.mlp`` with :class:`UntiedMLP`.
    2. Injects per-type QKV/output projections into the attention sub-layer
       **without disturbing its attention pattern**:

       - When the parent built a
         :class:`~noether.modeling.modules.attention.anchor_attention.multi_branch.MultiBranchAnchorAttention`
         (``SelfAnchorAttention`` / ``CrossAnchorAttention`` / ``JointAnchorAttention``),
         only its inner ``mixed_attention`` is swapped for
         :class:`UntiedMixedAttention`. The outer wrapper keeps emitting the
         correct per-branch attention patterns — so ``self_untied`` behaves
         like ``self`` with untied weights (not like a joint attention).
       - Otherwise (default ``dot_product`` attention, or any other non-multi-branch
         constructor) ``self.attention_block`` is replaced outright with
         :class:`UntiedMixedAttention`, which falls back to a single all-to-all
         pattern.

    3. Overrides :meth:`_mlp_forward` to route ``token_specs`` from
       ``attn_kwargs`` into :class:`UntiedMLP`.

    Attention receives ``token_specs`` automatically because the parent's
    forward passes ``**attn_kwargs`` to ``self.attention_block``, which in turn
    forwards them on to the inner :class:`UntiedMixedAttention`.

    Args:
        config: Configuration for the untied transformer block.
    """

    def __init__(self, config: UntiedTransformerBlockConfig) -> None:
        super().__init__(config=config.transformer_block)
        self.num_types = config.num_types
        # Replace the shared MLP with per-type weights.
        self.mlp = UntiedMLP(config=config.untied_mlp_config)  # type: ignore[arg-type,assignment]
        # Inject per-type QKV/output projections into the attention sub-layer.
        # For a MultiBranchAnchorAttention wrapper we swap only the inner
        # mixed_attention so that its pattern-emission logic is preserved.
        untied_attention = UntiedMixedAttention(config=config.attention_config)  # type: ignore[arg-type]
        if isinstance(self.attention_block, MultiBranchAnchorAttention):
            self.attention_block.mixed_attention = untied_attention
        else:
            self.attention_block = untied_attention  # type: ignore[assignment]

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
        attn_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]] | None]:
        """Validate ``token_specs`` upfront, then delegate to :meth:`TransformerBlock.forward`.

        The untied weight banks are indexed by name-order in ``token_specs``, so
        duplicates would silently merge types and missing types would index out
        of range. We check here (before attention runs) to surface a clear error.
        """
        if attn_kwargs is None or attn_kwargs.get("token_specs") is None:
            raise ValueError("attn_kwargs must be provided with 'token_specs' key for UntiedTransformerBlock.")
        token_specs = cast("Sequence[TokenSpec]", attn_kwargs["token_specs"])
        # _domain_group_sizes asserts that domains are strictly consecutive and
        # returns one size per domain. We further check that the number of
        # domains matches the weight bank count.
        domain_sizes = _domain_group_sizes(token_specs)
        assert len(domain_sizes) == self.num_types, (
            f"Number of domains in token_specs ({len(domain_sizes)}) must match num_types ({self.num_types})."
        )
        return super().forward(x, condition=condition, attn_kwargs=attn_kwargs)

    def _mlp_forward(self, x: torch.Tensor, attn_kwargs: dict[str, Any] | None = None) -> torch.Tensor:
        """Call :class:`UntiedMLP` with ``token_specs`` routed from ``attn_kwargs``.

        ``token_specs`` is guaranteed non-None here because :meth:`forward`
        validates upfront before delegating.
        """
        assert attn_kwargs is not None  # forward() has already validated this
        token_specs = cast("Sequence[TokenSpec]", attn_kwargs["token_specs"])
        out: torch.Tensor = self.mlp(x, token_specs)
        return out


class UntiedPerceiverBlock(PerceiverBlock):
    """Perceiver block with per-type (untied) Q/output projections and MLP weights.

    Same architecture and control flow as :class:`PerceiverBlock`; the Q and
    output projections in the attention module become per-type via
    :class:`UntiedPerceiverAttention`, and the feed-forward MLP becomes
    :class:`UntiedMLP`. The KV projection remains shared since it operates on
    a single source (e.g. geometry encoding).

    ``token_specs`` must be provided via ``attn_kwargs["token_specs"]``.

    Args:
        config: Configuration for the untied perceiver block.
    """

    def __init__(self, config: UntiedPerceiverBlockConfig) -> None:
        super().__init__(config=config.perceiver_block_config)
        self.num_types = config.num_types
        block = config.perceiver_block_config
        # Replace shared attention with per-type Q/output projections.
        self.attn = UntiedPerceiverAttention(  # type: ignore[assignment]
            config=block.perceiver_attention_config,  # type: ignore[arg-type]
            num_types=config.num_types,
        )
        # Replace shared MLP with per-type MLP.
        self.mlp = UntiedMLP(config=config.untied_mlp_config)  # type: ignore

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
        attn_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Forward pass with per-type Q/output projections and MLP.

        Validates ``token_specs`` upfront, then runs the same perceiver block
        logic as the parent with ``token_specs`` routed to the untied sub-modules.

        Args:
            q: Query tensor ``(B, S, hidden_dim)`` with tokens from all domains
                concatenated in ``token_specs`` order.
            kv: Key/value tensor from a single source (e.g. geometry encoding).
                Can be ``None`` when ``kv_cache`` is provided in ``attn_kwargs``.
            condition: Optional conditioning vector for modulation.
            attn_kwargs: Must contain ``"token_specs"`` key. Other entries
                (``kv_cache``, RoPE frequencies, masks) are forwarded to attention.

        Returns:
            Tuple of (output, kv_cache).
        """
        if attn_kwargs is None or attn_kwargs.get("token_specs") is None:
            raise ValueError("attn_kwargs must be provided with 'token_specs' key for UntiedPerceiverBlock.")
        token_specs = cast("Sequence[TokenSpec]", attn_kwargs["token_specs"])
        domain_sizes = _domain_group_sizes(token_specs)
        assert len(domain_sizes) == self.num_types, (
            f"Number of domains in token_specs ({len(domain_sizes)}) must match num_types ({self.num_types})."
        )

        # Separate token_specs from kwargs forwarded to attention.
        fwd_kwargs = {k: v for k, v in attn_kwargs.items() if k != "token_specs"}
        use_cached_kv = fwd_kwargs.get("kv_cache") is not None

        if self.modulation is None:
            if condition is not None:
                raise ValueError("Conditioning vector provided, but modulation is not configured.")
            attn_out, kv_cache_out = self.attn(
                q=self.norm1q(q), token_specs=token_specs, kv=self.norm1kv(kv) if kv is not None else None, **fwd_kwargs
            )
            q = q + self.drop_path1(self.ls1(attn_out))
            q = q + self.drop_path2(self.ls2(self.mlp(self.norm2(q), token_specs)))
        else:
            if condition is None:
                raise ValueError("No conditioning vector provided, but modulation is configured.")
            mod = self.modulation(condition)
            hd = self.norm1q.normalized_shape[0]
            kd = self._kv_dim
            q_scale, q_shift, kv_scale, kv_shift, attn_gate, mlp_scale, mlp_shift, mlp_gate = mod.split(
                [hd, hd, kd, kd, hd, hd, hd, hd], dim=-1
            )
            if use_cached_kv:
                normed_kv = None
            else:
                assert kv is not None, "kv must be provided when not using kv_cache"
                normed_kv = modulate_scale_shift(self.norm1kv(kv), scale=kv_scale, shift=kv_shift)

            attn_out, kv_cache_out = self.attn(
                q=modulate_scale_shift(self.norm1q(q), scale=q_scale, shift=q_shift),
                token_specs=token_specs,
                kv=normed_kv,
                **fwd_kwargs,
            )
            q = q + self.drop_path1(modulate_gate(self.ls1(attn_out), gate=attn_gate))
            q = q + self.drop_path2(
                modulate_gate(
                    self.ls2(
                        self.mlp(
                            modulate_scale_shift(self.norm2(q), scale=mlp_scale, shift=mlp_shift),
                            token_specs,
                        ),
                    ),
                    gate=mlp_gate,
                ),
            )
        return q, kv_cache_out
