#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any, Literal

import torch
from pydantic import BaseModel, Field, computed_field, model_validator
from torch import Tensor, nn

from noether.core.schemas.modules.attention import AttentionConfig
from noether.core.types import InitWeightsMode
from noether.modeling.functional.modulation import modulate_gate, modulate_scale_shift
from noether.modeling.modules.attention import ATTENTION_REGISTRY
from noether.modeling.modules.layers.drop_path import UnquantizedDropPath, UnquantizedDropPathConfig
from noether.modeling.modules.layers.layer_scale import LayerScale, LayerScaleConfig
from noether.modeling.modules.layers.linear_projection import LinearProjection, LinearProjectionConfig
from noether.modeling.modules.mlp.upactdown_mlp import UpActDownMlp, UpActDownMLPConfig


class TransformerBlockConfig(BaseModel):
    """Configuration for a transformer block."""

    hidden_dim: int = Field(..., ge=1)
    """Hidden Dimension of the transformer block."""

    num_heads: int = Field(..., ge=1)
    """Number of attention heads."""

    mlp_hidden_dim: int | None = Field(None)
    """Hidden dimension of the MLP layer. If set to None, the mlp_hidden dim is set to hidden_dim * mlp_expansion_factor in the TransformerConfig. If both are None, an error is raised."""

    mlp_expansion_factor: int | None = Field(None, ge=1)
    """Expansion factor for the MLP hidden dimension relative to the hidden dimension. If 'mlp_hidden_dim' is not set, this factor is used to compute it as hidden_dim * mlp_expansion_factor."""

    drop_path: float = Field(0.0, ge=0.0, le=1.0)
    """Probability to drop the attention or MLP module. Defaults to 0.0."""

    attention_constructor: Literal[
        "dot_product",
        "perceiver",
        "transolver",
        "transolver_plusplus",
    ] = "dot_product"
    """Constructor of the attention module. Defaults to 'dot_product'."""

    layerscale: float | None = Field(None, ge=0.0)
    """ Init scale value to scale layer activations. Defaults to None."""

    condition_dim: int | None = Field(None)
    """Dimension of the conditioning vector. If none, no conditioning is applied. If provided, the transformer block will turn into a Diffusion Transformer (DiT) block."""

    bias: bool = Field(True)
    """Whether to use biases in norm/projections. Defaults to True."""

    eps: float = Field(1e-6, gt=0.0)
    """Epsilon Value for the layer nornalization. Defaults to 1e-6."""

    init_weights: InitWeightsMode = Field("truncnormal002")
    """Initialization method for the weight matrices of the network. Defaults to "truncnormal002"""

    use_rope: bool = Field(False)
    """Whether to use Rotary Positional Embeddings (RoPE)."""

    max_wavelength: int | None = Field(10_000)
    """Theta parameter for the transformer sine/cosine embedding. Default: 10_000"""

    attention_arguments: dict = {}
    """Additional arguments for the attention module that are only needed for a specific attention implementation."""

    @model_validator(mode="after")
    def set_mlp_hidden_dim(self):
        # Validate hidden_dim is divisible by num_heads
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads}).")

        if self.mlp_hidden_dim is None:
            if self.mlp_expansion_factor is None:
                raise ValueError("Either 'mlp_hidden_dim' or 'mlp_expansion_factor' must be provided.")
            self.mlp_hidden_dim = self.hidden_dim * self.mlp_expansion_factor
        return self

    @model_validator(mode="after")
    def set_wavelength_for_rope(self):
        if self.use_rope and self.max_wavelength is None:
            raise ValueError("max_wavelength must be provided when use_rope is True.")
        return self

    @computed_field
    def linear_projection_config(self) -> "LinearProjectionConfig":
        return LinearProjectionConfig(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            bias=self.bias,
            init_weights=self.init_weights,
        )

    @computed_field
    def layerscale_config(self) -> "LayerScaleConfig":
        return LayerScaleConfig(
            hidden_dim=self.hidden_dim,
            init_values=self.layerscale,
        )

    @computed_field
    def drop_path_config(self) -> "UnquantizedDropPathConfig":
        return UnquantizedDropPathConfig(drop_prob=self.drop_path)

    @computed_field
    def modulation_linear_projection_config(self) -> "LinearProjectionConfig | None":
        if self.condition_dim is not None:
            return LinearProjectionConfig(
                input_dim=self.condition_dim,
                output_dim=self.hidden_dim * 6,
                init_weights="zeros",
            )
        return None

    @computed_field
    def up_act_down_mlp_config(self) -> "UpActDownMLPConfig":
        return UpActDownMLPConfig(
            input_dim=self.hidden_dim,
            hidden_dim=self.mlp_hidden_dim,
            bias=self.bias,
            init_weights=self.init_weights,
        )


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
