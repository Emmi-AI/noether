#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections.abc import Sequence

import einops
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from noether.core.schemas.modules import AttentionConfig, FLAREAttentionConfig
from noether.core.schemas.modules.attention import TokenSpec
from noether.modeling.functional.init import apply_init_method
from noether.modeling.modules.activations import Activation


class _ResidualMLP(nn.Module):
    """Residual MLP used by FLARE key/value projections."""

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        activation: str,
        bias: bool,
        init_weights: str,
        input_residual: bool,
        output_residual: bool,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.input_residual = input_residual and in_dim == hidden_dim
        self.output_residual = output_residual and hidden_dim == out_dim

        if num_layers == -1:
            self.fc = nn.Linear(in_dim, out_dim, bias=bias)
            self.residual = input_residual and output_residual and in_dim == out_dim
            apply_init_method(self, self.fc.weight, init_weights)
            return

        self.act = Activation[activation].build()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.fcs = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=bias) for _ in range(num_layers))
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=bias)
        apply_init_method(self, self.fc2.weight, init_weights)

    def forward(self, x: Tensor) -> Tensor:
        if self.num_layers == -1:
            y = self.fc(x)
            return x + y if self.residual else y

        y = self.act(self.fc1(x))
        x = x + y if self.input_residual else y
        for fc in self.fcs:
            x = x + self.act(fc(x))
        y = self.fc2(x)
        return x + y if self.output_residual else y


class FLAREAttention(nn.Module):
    """Fast Low-rank Attention Routing Engine.

    FLARE routes token information through per-head learned latent queries using
    two scaled-dot-product attention calls:

    ``latents = SDPA(Q_latent, K_tokens, V_tokens, scale=1.0)``
    ``tokens = SDPA(K_tokens, Q_latent, latents, scale=1.0)``

    This implementation follows the reference paper/code path while accepting
    Noether's usual attention kwargs. When ``token_specs`` contains groups whose
    names end with ``anchor_suffix``, only those tokens are used for encoding;
    all tokens are decoded. This preserves Noether/Terra anchor-query semantics.
    """

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        config = FLAREAttentionConfig(**config.model_dump())

        if config.use_rope:
            # FLARE does not apply RoPE internally; accepting `freqs` in forward
            # keeps it drop-in compatible with blocks that already compute RoPE.
            pass

        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.hidden_dim = config.hidden_dim
        self.num_latents = config.num_latents
        self.dropout = config.dropout
        self.attn_scale = config.attn_scale
        self.anchor_suffix = config.anchor_suffix

        self.latent_q = nn.Parameter(torch.empty(config.hidden_dim, config.num_latents))
        nn.init.normal_(self.latent_q, mean=0.0, std=config.latent_init_std)

        self.qk_norm = config.qk_norm
        if self.qk_norm:
            norm_cls = nn.RMSNorm if config.rmsnorm else nn.LayerNorm
            self.q_norm = norm_cls(self.head_dim)
            self.k_norm = norm_cls(self.head_dim)

        self.k_proj = _ResidualMLP(
            in_dim=config.hidden_dim,
            hidden_dim=max(1, int(config.hidden_dim * config.k_proj_mlp_ratio)),
            out_dim=config.hidden_dim,
            num_layers=config.num_layers_k_proj,
            activation=config.activation,
            bias=config.bias,
            init_weights=config.init_weights,
            input_residual=True,
            output_residual=True,
        )
        self.v_proj = _ResidualMLP(
            in_dim=config.hidden_dim,
            hidden_dim=max(1, int(config.hidden_dim * config.v_proj_mlp_ratio)),
            out_dim=config.hidden_dim,
            num_layers=config.num_layers_v_proj,
            activation=config.activation,
            bias=config.bias,
            init_weights=config.init_weights,
            input_residual=True,
            output_residual=True,
        )
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=config.bias)
        self.proj_dropout = nn.Dropout(config.dropout)
        apply_init_method(self, self.out_proj.weight, config.init_weights)

    @staticmethod
    def _normalize_token_mask(mask: Tensor, x: Tensor) -> Tensor:
        if mask.ndim == 4 and mask.shape[1:3] == (1, 1):
            mask = mask[:, 0, 0, :]
        elif mask.ndim == 3 and mask.shape[1] == 1:
            mask = mask[:, 0, :]
        if mask.ndim != 2:
            raise ValueError(
                f"FLAREAttention expects a bool token mask with shape (batch, tokens), got {tuple(mask.shape)}."
            )
        if mask.dtype != torch.bool:
            raise ValueError(f"FLAREAttention token mask must be bool, got {mask.dtype}.")
        if mask.shape != x.shape[:2]:
            raise ValueError(
                f"FLAREAttention token mask shape {tuple(mask.shape)} does not match "
                f"input token shape {tuple(x.shape[:2])}."
            )
        return mask

    def _anchor_mask_from_specs(
        self,
        token_specs: Sequence[TokenSpec] | None,
        x: Tensor,
    ) -> Tensor | None:
        if token_specs is None:
            return None

        pieces: list[Tensor] = []
        saw_anchor = False
        total = 0
        for spec in token_specs:
            if spec.size is None:
                raise ValueError("FLAREAttention does not support cached TokenSpec entries.")
            total += spec.size
            is_anchor = spec.name.endswith(self.anchor_suffix)
            saw_anchor = saw_anchor or is_anchor
            pieces.append(torch.full((spec.size,), is_anchor, dtype=torch.bool, device=x.device))

        if total != x.shape[1]:
            raise ValueError(f"TokenSpec sizes sum to {total}, but FLAREAttention received {x.shape[1]} tokens.")
        if not saw_anchor:
            return None
        return torch.cat(pieces).unsqueeze(0).expand(x.shape[0], -1)

    def _encode_mask(
        self,
        x: Tensor,
        *,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        token_specs: Sequence[TokenSpec] | None,
    ) -> Tensor | None:
        mask: Tensor | None = None
        if attn_mask is not None:
            mask = self._normalize_token_mask(attn_mask, x)
        if key_padding_mask is not None:
            key_padding_mask = self._normalize_token_mask(key_padding_mask, x)
            mask = key_padding_mask if mask is None else mask & key_padding_mask

        anchor_mask = self._anchor_mask_from_specs(token_specs, x)
        if anchor_mask is not None:
            mask = anchor_mask if mask is None else mask & anchor_mask
        return mask

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        token_specs: Sequence[TokenSpec] | None = None,
        freqs: Tensor | None = None,
        **_ignored,
    ) -> Tensor:
        """Apply FLARE attention.

        Args:
            x: Input tokens with shape ``(batch, tokens, hidden_dim)``.
            attn_mask: Optional bool token mask. ``True`` marks tokens that may
                participate in the latent encode step.
            key_padding_mask: Optional bool token mask using the same convention
                as ``attn_mask``. Provided for Noether anchor-attention parity.
            token_specs: Optional token group specs. Anchor groups are used for
                encoding when present; all tokens are decoded.
            freqs: Accepted for TransformerBlock compatibility. FLARE does not
                use RoPE internally.

        Returns:
            Tensor of shape ``(batch, tokens, hidden_dim)``.
        """
        del freqs

        q = self.latent_q.view(self.num_heads, self.num_latents, self.head_dim)
        k = einops.rearrange(
            self.k_proj(x),
            "batch tokens (heads dim) -> batch heads tokens dim",
            heads=self.num_heads,
        )
        v = einops.rearrange(
            self.v_proj(x),
            "batch tokens (heads dim) -> batch heads tokens dim",
            heads=self.num_heads,
        )

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        encode_mask = self._encode_mask(
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            token_specs=token_specs,
        )
        encode_attn_mask = encode_mask[:, None, None, :] if encode_mask is not None else None

        z = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=encode_attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.attn_scale,
        )
        y = F.scaled_dot_product_attention(
            k,
            q,
            z,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.attn_scale,
        )
        y = einops.rearrange(y, "batch heads tokens dim -> batch tokens (heads dim)")
        return self.proj_dropout(self.out_proj(y))
