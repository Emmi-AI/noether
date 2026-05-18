#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from noether.core.schemas.models import TransformerConfig, ViTConfig
from noether.core.schemas.modules import (
    ContinuousSincosEmbeddingConfig,
    RopeFrequencyConfig,
)
from noether.modeling.models.transformer import Transformer
from noether.modeling.modules.layers import (
    AvgPool2DPatchify,
    ContinuousSincosEmbed,
    ConvOutputHead,
    FinalLayer,
    MaskPatchify,
    RopeFrequency,
)


class ViT(nn.Module):
    """Vision Transformer for spatial regression on continuous-coordinate grids.

    Based on the ViT paper (https://arxiv.org/pdf/2010.11929) with several modifications, such as:

    - Continuous coordinate inputs with sincos positional embedding and RoPE (vs. learned 1D position embeddings).
    - Optional AdaLN-Zero conditioning, à la DiT (https://arxiv.org/abs/2212.09748).
    - RMSNorm and QK-norm in attention (vs. LayerNorm only).
    """

    def __init__(self, config: ViTConfig) -> None:
        """
        Args:
            config: Configuration for the ViT model. See
                :class:`~noether.core.schemas.models.ViTConfig` for available options.
        """
        super().__init__()

        self.coord_dim = config.coord_dim
        self.out_channels = config.out_channels
        self.patch_size = config.patch_size
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.token_dropout = config.token_dropout
        self.use_conditioning = config.use_conditioning

        # patchify
        self.pool_patch = AvgPool2DPatchify(patch_size=config.patch_size)
        self.mask_patchify = MaskPatchify(patch_size=config.patch_size)

        # positional encoding
        self.pos_embedding = ContinuousSincosEmbed(
            config=ContinuousSincosEmbeddingConfig(hidden_dim=config.hidden_dim, input_dim=config.coord_dim),  # type: ignore[call-arg]
        )
        self.rope = RopeFrequency(
            config=RopeFrequencyConfig(  # type: ignore[call-arg]
                input_dim=config.coord_dim,
                hidden_dim=config.hidden_dim // config.num_heads,
            ),
        )

        self.backbone = Transformer(
            config=TransformerConfig(
                name="vit_backbone",
                hidden_dim=config.hidden_dim,
                depth=config.depth,
                transformer_block_config=config.transformer_block_config,
            )
        )

        # output heads
        self.use_conv_output_head = config.use_conv_output_head
        if config.use_conv_output_head:
            self.final_layer = FinalLayer(
                config.hidden_dim, 1, config.hidden_dim, use_modulation=config.use_conditioning
            )
            self.conv_output_head: ConvOutputHead | None = ConvOutputHead(
                config.hidden_dim, config.out_channels, config.patch_size
            )
        else:
            self.final_layer = FinalLayer(
                config.hidden_dim, config.patch_size, config.out_channels, use_modulation=config.use_conditioning
            )
            self.conv_output_head = None

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize backbone weights"""
        if self.final_layer.adaLN_modulation is not None:
            nn.init.constant_(self.final_layer.adaLN_modulation.weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation.bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # Zero the last conv of the ConvOutputHead so decoding starts near identity.
        if self.conv_output_head is not None:
            last_stage = self.conv_output_head.stages[-1]
            if not isinstance(last_stage, nn.Sequential):
                raise ValueError("Expected last stage of ConvOutputHead to be nn.Sequential.")
            for module in reversed(list(last_stage)):
                if isinstance(module, nn.Conv2d):
                    nn.init.constant_(module.weight, 0)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                    break

    def unpatchify(self, x: Tensor, grid_h: int, grid_w: int) -> Tensor:
        """Linear unpatchify: ``(B, L, p²·C_out) → (B, H, W, C_out)``."""
        p = self.patch_size
        c = self.out_channels
        b, seq_len, patch_dim = x.shape
        if seq_len != grid_h * grid_w:
            raise ValueError(f"Sequence length {seq_len} doesn't match grid {grid_h}*{grid_w}.")
        if patch_dim != p * p * c:
            raise ValueError(f"Patch dim mismatch: expected {p * p * c}, got {patch_dim}.")
        x = x.view(b, grid_h, grid_w, p, p, c)
        return rearrange(x, "b h w p q c -> b (h p) (w q) c")

    def forward(
        self,
        x: Tensor | None,
        coords: Tensor,
        mask: Tensor | None = None,
        cond: Tensor | None = None,
        return_tokens: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[int, int]]:
        """Run the standard ViT.

        Args:
            x: Optional pre-computed patch embeddings of shape ``(B, L, hidden_dim)``. When
                ``None``, tokens come purely from positional encoding.
            coords: Per-cell coordinates of shape ``(B, H, W, coord_dim)``.
            mask: Optional per-cell fluid mask of shape ``(B, H, W)``.
            cond: AdaLN conditioning vector of shape ``(B, hidden_dim)``. Required when the ViT
                was built with ``use_conditioning=True`` (the default); must be ``None`` otherwise.
            return_tokens: If True, return raw post-FinalLayer tokens plus ``(grid_h, grid_w)``
                instead of the decoded spatial output.

        Returns:
            Either ``(B, H, W, out_channels)`` or ``(tokens, (grid_h, grid_w))`` if
            ``return_tokens``.
        """
        if self.use_conditioning and cond is None:
            raise ValueError("ViT was built with use_conditioning=True; `cond` is required.")
        if not self.use_conditioning and cond is not None:
            raise ValueError("ViT was built with use_conditioning=False; `cond` must be None.")

        # Patchify coords (used for both sincos pos embed and RoPE).
        coords_patched = self.pool_patch(coords)  # (B, gh, gw, coord_dim)
        _, grid_h, grid_w, _ = coords_patched.shape
        coords_flat = coords_patched.flatten(1, 2)  # (B, L, coord_dim)

        rope_freqs = self.rope(coords_flat)
        pos_encoded = self.pos_embedding(coords_flat)
        tokens = pos_encoded

        if x is not None:
            tokens = tokens + x

        patch_mask: Tensor | None = None
        if mask is not None:
            patch_mask = self.mask_patchify(mask)
            tokens = tokens * patch_mask.unsqueeze(-1).float()

        condition = F.silu(cond) if cond is not None else None

        attn_kwargs = {"freqs": rope_freqs}
        if patch_mask is not None:
            # Patch mask is also attention mask; (B, L) -> (B, 1, 1, L) so SDPA broadcasts across heads and queries
            attn_kwargs["attn_mask"] = patch_mask[:, None, None, :]

        tokens = self.backbone(tokens, attn_kwargs=attn_kwargs, condition=condition)

        tokens = self.final_layer(tokens, condition)

        if return_tokens:
            return tokens, (grid_h, grid_w)

        if self.conv_output_head is not None:
            decoded: Tensor = self.conv_output_head(tokens, grid_h, grid_w)
            return decoded

        return self.unpatchify(tokens, grid_h, grid_w)
