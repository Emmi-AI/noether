#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from noether.core.schemas.modules import AttentionConfig, LinearProjectionConfig, TransolverAttentionConfig
from noether.modeling.modules.layers import LinearProjection


class TransolverAttention(nn.Module):
    """Adapted from https://github.com/thuml/Transolver/blob/main/Car-Design-ShapeNetCar/models/Transolver.py
    - Readable reshaping operations via einops
    - Merged qkv linear layer for higher GPU utilization
    - F.scaled_dot_product_attention instead of slow pytorch attention
    - Possibility to mask tokens (required to process variable sized inputs)"""

    def __init__(self, config: AttentionConfig):
        """
        Initialize the Transolver attention module.

        Args:
            config: Configuration for the Transolver attention module. See
                :class:`~noether.core.schemas.modules.AttentionConfig` for available options.
        """

        super().__init__()
        config = TransolverAttentionConfig(**config.model_dump())

        dim_head = config.hidden_dim // config.num_heads
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        self.temperature = nn.Parameter(torch.full(size=(1, config.num_heads, 1, 1), fill_value=0.5))

        self.in_project_x = LinearProjection(
            config=LinearProjectionConfig(
                input_dim=config.hidden_dim,
                output_dim=config.hidden_dim,
                init_weights=config.init_weights,
                bias=config.bias,
            )  # type: ignore[call-arg]
        )
        self.in_project_fx = LinearProjection(
            config=LinearProjectionConfig(
                input_dim=config.hidden_dim,
                output_dim=config.hidden_dim,
                init_weights=config.init_weights,
                bias=config.bias,
            )  # type: ignore[call-arg]
        )
        self.in_project_slice = LinearProjection(
            config=LinearProjectionConfig(
                input_dim=dim_head, output_dim=config.num_slices, init_weights=config.init_weights, bias=config.bias
            )  # type: ignore[call-arg]
        )

        self.qkv = LinearProjection(
            config=LinearProjectionConfig(
                input_dim=dim_head,
                output_dim=dim_head * 3,
                bias=False,
                init_weights=config.init_weights,
            )  # type: ignore[call-arg]
        )
        self.proj = LinearProjection(
            config=LinearProjectionConfig(
                input_dim=config.hidden_dim,
                output_dim=config.hidden_dim,
                bias=config.bias,
                init_weights=config.init_weights,
            )  # type: ignore[call-arg]
        )
        self.proj_dropout = nn.Dropout(config.dropout)

    def create_slices(self, x: torch.Tensor, num_input_points: int, attn_mask: torch.Tensor | None = None):
        """Given a set of points, project them to a fixed number of slices using the computed the slice weights per token.

        Args:
            x: Input tensor with shape (batch_size, num_input_points, hidden_dim).
            num_input_points: Number of input points.
            attn_mask: Mask to mask out certain token for the attention. Defaults to None.

        Returns:
            Tensor with the projected slice tokens and the slice weights.
        """

        # slice - project the input points to a fixed number of physics tokens/slices
        fx_mid = einops.rearrange(
            self.in_project_fx(x),
            "batch_size num_points (num_heads dim_head) -> batch_size num_heads num_points dim_head",
            num_heads=self.num_heads,
        ).contiguous()

        x_mid = einops.rearrange(
            self.in_project_x(x),
            "batch_size num_points (num_heads dim_head) -> batch_size num_heads num_points dim_head",
            num_heads=self.num_heads,
        ).contiguous()

        slice_weights = F.softmax(self.in_project_slice(x_mid) / self.temperature, dim=-1)
        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool, "only bool mask supported"
            assert attn_mask.ndim == 2
            assert len(attn_mask) == len(x)
            assert attn_mask.size(1) == num_input_points

            attn_mask = einops.rearrange(attn_mask, "batch_size num_points -> batch_size 1 num_points 1").float()
            slice_weights = slice_weights * attn_mask

        slice_norm = einops.rearrange(
            slice_weights.sum(2),
            "batch_size num_heads num_slices -> batch_size num_heads num_slices 1",
        )
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights) / (slice_norm + 1e-5)

        return slice_token, slice_weights

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
        """Forward pass of the Transolver attention module.

        Args:
            x: Input tensor with shape (batch_size, seqlen, hidden_dim).
            attn_mask: Attention mask tensor with shape (batch_size). Defaults to None.

        Returns:
            Tensor after applying the transolver attention mechanism.
        """

        _, num_input_points, _ = x.shape

        slice_token, slice_weights = self.create_slices(x, num_input_points=num_input_points, attn_mask=attn_mask)

        # attention among slice tokens
        q_slice_token, k_slice_token, v_slice_token = self.qkv(slice_token).chunk(3, dim=-1)
        out_slice_token = F.scaled_dot_product_attention(
            q_slice_token,
            k_slice_token,
            v_slice_token,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # deslice - project the slice tokens back to the original points
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = einops.rearrange(
            out_x,
            "batch_size num_heads num_points dim_head -> batch_size num_points (num_heads dim_head)",
        )
        return self.proj_dropout(self.proj(out_x))
