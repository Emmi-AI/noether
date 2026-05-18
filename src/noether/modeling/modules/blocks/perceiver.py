#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

import torch
from torch import Tensor, nn

from noether.core.schemas.modules.blocks import PerceiverBlockConfig
from noether.modeling.functional.modulation import modulate_gate, modulate_scale_shift
from noether.modeling.modules.attention import PerceiverAttention
from noether.modeling.modules.layers import LayerScale, LinearProjection, UnquantizedDropPath
from noether.modeling.modules.mlp import UpActDownMlp


class PerceiverBlock(nn.Module):
    """For a self-attention module, the input tensor for the query, key, and value are the same. The PerceiverBlock,
    takes different input tensors for the query and the key/value.
    """

    def __init__(
        self,
        config: PerceiverBlockConfig,
    ):
        """

        Args:
            config: Configuration of the PerceiverBlock. See :class:`~noether.core.schemas.modules.blocks.PerceiverBlockConfig`
            for available options.
        """
        super().__init__()

        # modulation
        if config.condition_dim is None:
            self.modulation = None
            elementwise_affine = True
        else:
            self._kv_dim = config.kv_dim or config.hidden_dim
            if config.modulation_linear_projection_config is not None:
                self.modulation = LinearProjection(config=config.modulation_linear_projection_config)  # type: ignore[arg-type]
                elementwise_affine = False
            else:
                raise ValueError(
                    "If modulation is enabled, modulation_linear_projection_config must be provided. Likely condition_dim is not set."
                )

        self.norm1q = nn.RMSNorm(config.hidden_dim, eps=config.eps, elementwise_affine=elementwise_affine)
        self.norm1kv = nn.RMSNorm(
            config.kv_dim or config.hidden_dim, eps=config.eps, elementwise_affine=elementwise_affine
        )

        self.attn = PerceiverAttention(config=config.perceiver_attention_config)  # type: ignore[arg-type]

        self.ls1 = LayerScale(config=config.layerscale_config)  # type: ignore[arg-type]

        self.drop_path1 = UnquantizedDropPath(config=config.drop_path_config)  # type: ignore[arg-type]

        self.norm2 = nn.RMSNorm(config.hidden_dim, eps=config.eps, elementwise_affine=elementwise_affine)

        self.mlp = UpActDownMlp(config=config.up_act_down_mlp_config)  # type: ignore[arg-type]
        self.ls2 = LayerScale(config=config.layerscale_config)  # type: ignore[arg-type]
        self.drop_path2 = UnquantizedDropPath(config=config.drop_path_config)  # type: ignore[arg-type]

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
        attn_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor] | None]:
        """Forward pass of the PerceiverBlock.

        Args:
            q: Input tensor with shape (batch_size, seqlen/num_tokens, hidden_dim) for the query representations.
            kv: Input tensor with shape (batch_size, seqlen/num_tokens, hidden_dim) for the key and value representations.
                Can be ``None`` when a ``kv_cache`` is provided in ``attn_kwargs``.
            condition: Conditioning vector. If provided, the attention and MLP will be scaled, shifted and gated
                feature-wise with predicted values from this vector.
            attn_kwargs: Dict with arguments for the attention (such as the attention mask, rope frequencies,
                or kv_cache). Defaults to None.

        Returns:
            Tuple of (output_tensor, kv_cache). ``kv_cache`` contains cached K/V from the
            perceiver attention, or ``None`` when loading from cache.
        """
        use_cached_kv = attn_kwargs is not None and attn_kwargs.get("kv_cache") is not None

        if self.modulation is None:
            if condition is not None:
                raise ValueError("Conditioning vector provided, but modulation is not configured.")
            attn_out, kv_cache_out = self.attn(
                q=self.norm1q(q), kv=self.norm1kv(kv) if kv is not None else None, **(attn_kwargs or {})
            )
            q = q + self.drop_path1(self.ls1(attn_out))
            q = q + self.drop_path2(self.ls2(self.mlp(self.norm2(q))))
        else:
            if condition is None:
                raise ValueError("No conditioning vector provided, but modulation is configured.")
            mod = self.modulation(condition)
            hd = self.norm1q.normalized_shape[0]
            kd = self._kv_dim
            q_scale, q_shift, kv_scale, kv_shift, attn_gate, mlp_scale, mlp_shift, mlp_gate = mod.split(
                [hd, hd, kd, kd, hd, hd, hd, hd], dim=-1
            )
            # In cached mode, kv is None — skip normalization and modulation of kv
            if use_cached_kv:
                normed_kv = None
            else:
                assert kv is not None, "kv must be provided when not using kv_cache"
                normed_kv = modulate_scale_shift(self.norm1kv(kv), scale=kv_scale, shift=kv_shift)

            attn_out, kv_cache_out = self.attn(
                q=modulate_scale_shift(self.norm1q(q), scale=q_scale, shift=q_shift),
                kv=normed_kv,
                **(attn_kwargs or {}),
            )
            q = q + self.drop_path1(
                modulate_gate(
                    self.ls1(attn_out),
                    gate=attn_gate,
                ),
            )
            q = q + self.drop_path2(
                modulate_gate(
                    self.ls2(
                        self.mlp(
                            modulate_scale_shift(
                                self.norm2(q),
                                scale=mlp_scale,
                                shift=mlp_shift,
                            ),
                        ),
                    ),
                    gate=mlp_gate,
                ),
            )
        return q, kv_cache_out
