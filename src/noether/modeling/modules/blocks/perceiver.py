#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

import torch
from torch import nn

from noether.core.schemas.modules.attention import PerceiverAttentionConfig
from noether.core.schemas.modules.blocks import PerceiverBlockConfig
from noether.core.schemas.modules.layers import LayerScaleConfig, LinearProjectionConfig, UnquantizedDropPathConfig
from noether.core.schemas.modules.mlp import UpActDownMLPConfig
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
            assert config.kv_dim is None
            assert config.bias
            self.modulation = LinearProjection(
                config=LinearProjectionConfig(
                    input_dim=config.condition_dim,
                    output_dim=config.hidden_dim * 8,
                    init_weights="zeros",
                )  # type: ignore[call-arg]
            )
            elementwise_affine = False

        self.norm1q = config.normalization_constructor(
            config.hidden_dim, elementwise_affine=elementwise_affine, bias=config.bias, eps=config.eps
        )
        self.norm1kv = config.normalization_constructor(
            config.kv_dim or config.hidden_dim, elementwise_affine=elementwise_affine, bias=config.bias, eps=config.eps
        )
        self.attn = PerceiverAttention(
            config=PerceiverAttentionConfig(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                kv_dim=config.kv_dim,
                bias=config.bias,
                init_weights=config.init_weights,
                use_rope=config.use_rope,
            )  # type: ignore[call-arg]
        )
        self.ls1 = LayerScale(config=LayerScaleConfig(hidden_dim=config.hidden_dim, init_values=config.layerscale))
        self.drop_path1 = UnquantizedDropPath(
            config=UnquantizedDropPathConfig(drop_prob=config.drop_path)  # type: ignore[call-arg]
        )

        self.norm2 = config.normalization_constructor(
            config.hidden_dim, elementwise_affine=elementwise_affine, bias=config.bias, eps=config.eps
        )

        self.mlp = UpActDownMlp(
            config=UpActDownMLPConfig(
                input_dim=config.hidden_dim,
                hidden_dim=config.mlp_hidden_dim or config.hidden_dim * 4,
                bias=config.bias,
                init_weights=config.init_weights,
            )  # type: ignore[call-arg]
        )
        self.ls2 = LayerScale(config=LayerScaleConfig(hidden_dim=config.hidden_dim, init_values=config.layerscale))
        self.drop_path2 = UnquantizedDropPath(
            config=UnquantizedDropPathConfig(drop_prob=config.drop_path)  # type: ignore[call-arg]
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        condition: torch.Tensor | None = None,
        attn_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Forward pass of the PerceiverBlock.

        Args:
            q: Input tensor with shape (batch_size, seqlen/num_tokens, hidden_dim) for the query representations.
            kv: Input tensor with shape (batch_size, seqlen/num_tokens, hidden_dim) for the key and value representations.
            condition: Conditioning vector. If provided, the attention and MLP will be scaled, shifted and gated
                feature-wise with predicted values from this vector.
            attn_kwargs: Dict with arguments for the attention (such as the attention mask or rope frequencies). Defaults to None.

        Returns:
            Tensor after the forward pass of the PerceiverBlock.
        """
        if self.modulation is None:
            if condition is not None:
                raise ValueError("Conditioning vector provided, but modulation is not configured.")
            q = q + self.drop_path1(self.ls1(self.attn(q=self.norm1q(q), kv=self.norm1kv(kv), **(attn_kwargs or {}))))
            q = q + self.drop_path2(self.ls2(self.mlp(self.norm2(q))))
        else:
            if condition is None:
                raise ValueError("No conditioning vector provided, but modulation is configured.")
            mod = self.modulation(condition)
            q_scale, q_shift, kv_scale, kv_shift, attn_gate, mlp_scale, mlp_shift, mlp_gate = mod.chunk(8, dim=-1)
            q = q + self.drop_path1(
                modulate_gate(
                    self.ls1(
                        self.attn(
                            q=modulate_scale_shift(self.norm1q(q), scale=q_scale, shift=q_shift),
                            kv=modulate_scale_shift(self.norm1kv(kv), scale=kv_scale, shift=kv_shift),
                            **(attn_kwargs or {}),
                        ),
                    ),
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
        return q
