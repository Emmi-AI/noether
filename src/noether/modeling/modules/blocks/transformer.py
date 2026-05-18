#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

import torch
from torch import Tensor, nn

from noether.core.schemas.modules.attention import AttentionConfig
from noether.core.schemas.modules.blocks import TransformerBlockConfig
from noether.modeling.functional.modulation import modulate_gate, modulate_scale_shift
from noether.modeling.modules.attention import ATTENTION_REGISTRY
from noether.modeling.modules.layers import LayerScale, LinearProjection, UnquantizedDropPath
from noether.modeling.modules.mlp import UpActDownMlp


class TransformerBlock(nn.Module):
    """A transformer block with a single attention layer and a feedforward layer."""

    def __init__(
        self,
        config: TransformerBlockConfig,
    ):
        """

        Args:
            config: Configuration for the transformer block. See
                :class:`~noether.core.schemas.modules.blocks.TransformerBlockConfig`
                for available options.
        """
        super().__init__()
        self.config = config
        # modulation
        if config.condition_dim is None:
            self.modulation = None
            elementwise_affine = True
        else:
            if config.modulation_linear_projection_config is None:
                raise ValueError("modulation_linear_projection_config must be provided if condition_dim is not None.")

            self.modulation = LinearProjection(config=config.modulation_linear_projection_config)  # type: ignore[arg-type]
            elementwise_affine = False

        self.norm1 = nn.RMSNorm(config.hidden_dim, eps=config.eps, elementwise_affine=elementwise_affine)

        try:
            if callable(config.attention_constructor):
                attention_class = config.attention_constructor
            else:
                attention_class = ATTENTION_REGISTRY[config.attention_constructor]
        except KeyError as exc:
            raise ValueError(
                f"Unknown attention_constructor='{config.attention_constructor}'. "
                f"Available: {sorted(ATTENTION_REGISTRY.keys())}"
            ) from exc

        self.attention_block = attention_class(
            config=AttentionConfig(
                **config.model_dump(),
                **(config.attention_arguments or {}),
            )
        )
        self.ls1 = LayerScale(config=config.layerscale_config)  # type: ignore[arg-type]
        self.drop_path1 = UnquantizedDropPath(
            config=config.drop_path_config  # type: ignore[arg-type]
        )
        self.norm2 = nn.RMSNorm(config.hidden_dim, eps=config.eps, elementwise_affine=elementwise_affine)
        self.mlp = UpActDownMlp(config=config.up_act_down_mlp_config)  # type: ignore[arg-type]
        self.ls2 = LayerScale(config=config.layerscale_config)  # type: ignore[arg-type]
        self.drop_path2 = UnquantizedDropPath(config=config.drop_path_config)  # type: ignore[arg-type]

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
        attn_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Tensor, dict[str, dict[str, Tensor]] | None]:
        """Forward pass of the transformer block.

        Args:
            x: Input tensor with shape (batch_size, seqlen/num_tokens, hidden_dim).
            condition: Conditioning vector. If provided, the attention and MLP will be scaled, shifted and gated
                feature-wise with predicted values from this vector.
            attn_kwargs: Dict with arguments for the attention (such as the attention mask or rope frequencies). Defaults to None.

        Returns:
            Tuple of (output_tensor, kv_cache). ``kv_cache`` is ``None`` when the attention module
            does not return a cache (e.g. standard ``DotProductAttention``).
        """
        if self.modulation is None:
            if condition is not None:
                raise ValueError(
                    "Conditioning vector provided, but the transformer block is not configured for conditioning."
                )
            attn_out = self.attention_block(self.norm1(x), **(attn_kwargs or {}))
            kv_cache = None
            if isinstance(attn_out, tuple):
                attn_out, kv_cache = attn_out
            x = x + self.drop_path1(self.ls1(attn_out))
            x = x + self.drop_path2(self.ls2(self._mlp_forward(self.norm2(x), attn_kwargs=attn_kwargs)))
        else:
            if condition is None:
                raise ValueError(
                    "No conditioning vector provided, but the transformer block is configured for conditioning."
                )
            if condition.shape[-1] != self.config.condition_dim:
                raise ValueError(
                    f"Conditioning vector has incorrect shape. Expected {self.config.condition_dim}, got {condition.shape[-1]}"
                )

            mod = self.modulation(condition)
            attn_scale, attn_shift, attn_gate, mlp_scale, mlp_shift, mlp_gate = mod.chunk(6, dim=-1)
            attn_out = self.attention_block(
                modulate_scale_shift(self.norm1(x), scale=attn_scale, shift=attn_shift),
                **(attn_kwargs or {}),
            )
            kv_cache = None
            if isinstance(attn_out, tuple):
                attn_out, kv_cache = attn_out
            x = x + self.drop_path1(
                modulate_gate(
                    self.ls1(attn_out),
                    gate=attn_gate,
                ),
            )
            x = x + self.drop_path2(
                modulate_gate(
                    self.ls2(
                        self._mlp_forward(
                            modulate_scale_shift(self.norm2(x), scale=mlp_scale, shift=mlp_shift),
                            attn_kwargs=attn_kwargs,
                        )
                    ),
                    gate=mlp_gate,
                ),
            )
        return x, kv_cache

    def _mlp_forward(self, x: torch.Tensor, attn_kwargs: dict[str, Any] | None = None) -> torch.Tensor:
        """Apply the MLP sub-layer.

        Override in subclasses that need to pass extra arguments to ``self.mlp``
        (for example, a token-spec-aware MLP that must route tokens to per-type
        weight banks). The default implementation ignores ``attn_kwargs`` and
        calls ``self.mlp`` with just the input tensor.

        Args:
            x: Input to the MLP, shape ``(B, S, hidden_dim)``.
            attn_kwargs: Same dict passed to :meth:`forward`; subclasses may read
                extra keys (e.g., ``token_specs``) from it.

        Returns:
            MLP output, shape ``(B, S, hidden_dim)``.
        """
        return self.mlp(x)  # type: ignore[no-any-return]
