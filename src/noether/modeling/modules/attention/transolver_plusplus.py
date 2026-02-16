#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from noether.core.schemas.modules import AttentionConfig, LinearProjectionConfig, TransolverPlusPlusAttentionConfig
from noether.modeling.modules.activations import Activation
from noether.modeling.modules.layers import LinearProjection


def _gumbel_softmax(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """Sample from the Gumbel-Softmax distribution.

    Lower temperatures corresponds to more discrete (one-hot) samples, higher temperatures lead to uniform sampling.
    See also torch.nn.functional.gumbel_softmax (deprecated), which has a straight-through estimator option.

    Args:
        logits: Logits for the Gumbel-Softmax distribution. Tensor, where the last dimension is the number of classes.
        temperature: Temperature param for the Gumbel-Softmax distribution. Scalar or tensor of same shape as logits.
    Returns:
        Sampled tensor from the Gumbel-Softmax distribution.
    """

    eps = torch.finfo(logits.dtype).eps  # machine epsilon for stable logarithms
    u = torch.rand_like(logits)
    gumbel_samples = -torch.log(-torch.log(u + eps) + eps)

    return F.softmax((logits + gumbel_samples) / temperature, dim=-1)


class TransolverPlusPlusAttention(nn.Module):
    """
    Transolver++ Attention module as implemented in https://github.com/thuml/Transolver_plus/blob/main/models/Transolver_plus.py
    """

    def __init__(self, config: AttentionConfig):
        """

        Args:
            config: Configuration for the TransolverPlusPlusAttention module. See
                :class:`~noether.core.schemas.modules.AttentionConfig` for available options.
        """
        super().__init__()

        config = TransolverPlusPlusAttentionConfig(**config.model_dump())

        inner_dim = config.head_dim * config.num_heads  # type: ignore[operator]
        self.dim_head = config.head_dim
        self.num_heads = config.num_heads
        self.scale = self.dim_head**-0.5  # type: ignore[operator]
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = config.dropout
        self.bias = nn.Parameter(torch.ones([1, self.num_heads, 1, 1]) * 0.5)

        self.proj_temperature = nn.Sequential(
            LinearProjection(
                config=LinearProjectionConfig(
                    input_dim=self.dim_head,  # type: ignore[arg-type]
                    output_dim=config.num_slices,
                    init_weights=config.init_weights,
                    bias=config.bias,
                )  # type: ignore[call-arg]
            ),
            Activation.GELU.value,
            LinearProjection(
                config=LinearProjectionConfig(
                    input_dim=config.num_slices, output_dim=1, init_weights=config.init_weights, bias=config.bias
                )  # type: ignore[call-arg]
            ),
            Activation.GELU.value,
        )

        self.in_project_x = nn.Linear(config.hidden_dim, inner_dim)
        self.in_project_slice = LinearProjection(
            config=LinearProjectionConfig(
                input_dim=self.dim_head,  # type: ignore[arg-type]
                output_dim=config.num_slices,
                init_weights=config.init_weights,
                bias=config.bias,
            )  # type: ignore[call-arg]
        )

        if isinstance(self.in_project_slice.project, nn.Linear | nn.Conv1d | nn.Conv2d | nn.Conv3d):
            torch.nn.init.orthogonal_(self.in_project_slice.project.weight)  # use a principled initialization

        self.qkv = LinearProjection(
            config=LinearProjectionConfig(
                input_dim=self.dim_head,  # type: ignore[arg-type]
                output_dim=self.dim_head * 3,  # type: ignore[operator]
                init_weights=config.init_weights,
                bias=config.bias,
            )  # type: ignore[call-arg]
        )

        self.to_out = nn.Sequential(
            LinearProjection(
                config=LinearProjectionConfig(
                    input_dim=inner_dim,
                    output_dim=config.hidden_dim,
                    init_weights=config.init_weights,
                    bias=config.bias,
                )  # type: ignore[call-arg]
            ),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
        """Forward pass of the Transolver attention module.

        Args:
            x: Input tensor with shape (batch_size, seqlen, hidden_dim).
            attn_mask: Attention mask tensor with shape (batch_size). Defaults to None.

        Returns:
            Tensor after applying the transolver attention mechanism.
        """

        batch_size, num_input_points, _ = x.shape

        x_mid = (
            self.in_project_x(x)
            .reshape(batch_size, num_input_points, self.num_heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # B H N C

        temperature = self.proj_temperature(x_mid) + self.bias
        temperature = torch.clamp(temperature, min=0.01)
        slice_weights = _gumbel_softmax(self.in_project_slice(x_mid), temperature)
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", x_mid, slice_weights).contiguous()
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))  # type: ignore[arg-type]

        q_slice_token, k_slice_token, v_slice_token = self.qkv(slice_token).chunk(3, dim=-1)
        out_slice_token = F.scaled_dot_product_attention(
            q_slice_token, k_slice_token, v_slice_token, dropout_p=self.dropout if self.training else 0.0
        )

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)
