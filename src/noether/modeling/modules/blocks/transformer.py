#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

import torch
from torch import nn

from noether.core.schemas.modules.attention import AttentionConfig
from noether.core.schemas.modules.blocks import TransformerBlockConfig
from noether.core.schemas.modules.layers import LayerScaleConfig, LinearProjectionConfig, UnquantizedDropPathConfig
from noether.core.schemas.modules.mlp import UpActDownMLPConfig
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
        """Initializes a transformer block.

        Args:
            config: Configuration for the transformer block. See
                :class:`~noether.core.schemas.modules.blocks.TransformerBlockConfig`
                for available options.
        """
        super().__init__()
        # modulation
        if config.condition_dim is None:
            self.modulation = None
            elementwise_affine = True
        else:
            assert config.bias
            self.modulation = LinearProjection(
                config=LinearProjectionConfig(
                    input_dim=config.condition_dim, output_dim=config.hidden_dim * 6, init_weights="zeros"
                )  # type: ignore[call-arg]
            )
            elementwise_affine = False

        self.norm1 = config.normalization_constructor(
            config.hidden_dim, elementwise_affine=elementwise_affine, bias=config.bias, eps=config.eps
        )

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
                hidden_dim=config.mlp_hidden_dim,  # type: ignore[arg-type]
                bias=config.bias,
                init_weights=config.init_weights,
            )
        )
        self.ls2 = LayerScale(config=LayerScaleConfig(hidden_dim=config.hidden_dim, init_values=config.layerscale))
        self.drop_path2 = UnquantizedDropPath(
            config=UnquantizedDropPathConfig(drop_prob=config.drop_path)  # type: ignore[call-arg]
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
        attn_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Forward pass of the transformer block.

        Args:
            x: Input tensor with shape (batch_size, seqlen/num_tokens, hidden_dim).
            condition: Conditioning vector. If provided, the attention and MLP will be scaled, shifted and gated
                feature-wise with predicted values from this vector.
            attn_kwargs: Dict with arguments for the attention (such as the attention mask). Defaults to None.

        Returns:
            Tensor after the forward pass of the transformer block.
        """
        if self.modulation is None:
            if condition is not None:
                raise ValueError(
                    "Conditioning vector provided, but the transformer block is not configured for conditioning."
                )
            x = x + self.drop_path1(self.ls1(self.attention_block(self.norm1(x), **(attn_kwargs or {}))))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        else:
            if condition is None:
                raise ValueError(
                    "No conditioning vector provided, but the transformer block is configured for conditioning."
                )

            mod = self.modulation(condition)
            attn_scale, attn_shift, attn_gate, mlp_scale, mlp_shift, mlp_gate = mod.chunk(6, dim=-1)
            x = x + self.drop_path1(
                modulate_gate(
                    self.ls1(
                        self.attention_block(
                            modulate_scale_shift(self.norm1(x), scale=attn_scale, shift=attn_shift),
                            **(attn_kwargs or {}),
                        ),
                    ),
                    gate=attn_gate,
                ),
            )
            x = x + self.drop_path2(
                modulate_gate(
                    self.ls2(self.mlp(modulate_scale_shift(self.norm2(x), scale=mlp_scale, shift=mlp_shift))),
                    gate=mlp_gate,
                ),
            )
        return x
