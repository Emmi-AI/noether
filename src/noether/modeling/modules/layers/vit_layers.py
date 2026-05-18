#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from noether.modeling.functional.modulation import modulate_scale_shift


class AvgPool2DPatchify(nn.Module):
    """Tokenize a 2D grid by average-pooling each ``patch_size``×``patch_size`` patch."""

    def __init__(self, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        """Pool spatial features into patches.

        Args:
            x: Input grid with shape ``(B, H, W, C)``.

        Returns:
            Pooled patch grid of shape ``(B, H // patch_size, W // patch_size, C)``.
        """
        x = rearrange(x, "b h w c -> b c h w")
        x = self.patch(x)
        return rearrange(x, "b c h w -> b h w c")


class MaskPatchify(nn.Module):
    """Downsample a boolean mask to patch resolution via max-pooling (``True`` = at least one valid cell)."""

    def __init__(self, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size

    def forward(self, mask: Tensor) -> Tensor:
        """Downsample boolean mask to patch resolution.

        Args:
            mask: Boolean mask of shape ``(B, H, W)``.

        Returns:
            Flat boolean mask of shape ``(B, (H // patch_size) * (W // patch_size))``.
        """
        pooled = F.max_pool2d(mask.float(), kernel_size=self.patch_size, stride=self.patch_size)
        return pooled.flatten(1).bool()


class FinalLayer(nn.Module):
    """Final unpatchify projection with optional AdaLN modulation conditioned on a global vector ``c``."""

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        use_modulation: bool = True,
    ) -> None:
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, eps=1e-6, elementwise_affine=True)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation: nn.Linear | None = (
            nn.Linear(hidden_size, 2 * hidden_size, bias=True) if use_modulation else None
        )

    def forward(self, x: Tensor, c: Tensor | None = None) -> Tensor:
        """Apply (optionally AdaLN-modulated) norm then linear projection.

        Args:
            x: Tokens of shape ``(B, L, hidden_size)``.
            c: Conditioning vector of shape ``(B, hidden_size)`` when ``use_modulation=True``;
                must be ``None`` when ``use_modulation=False``. The caller is responsible for any
                upstream activation (e.g. SiLU) — this layer applies the AdaLN linear directly.

        Returns:
            Tensor of shape ``(B, L, patch_size**2 * out_channels)``.
        """
        if self.adaLN_modulation is None:
            if c is not None:
                raise ValueError("FinalLayer was built with use_modulation=False; do not pass `c`.")
            return self.linear(self.norm_final(x))  # type: ignore[no-any-return]
        if c is None:
            raise ValueError("FinalLayer was built with use_modulation=True; a conditioning vector `c` is required.")
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate_scale_shift(self.norm_final(x), scale=scale, shift=shift)
        return self.linear(x)  # type: ignore[no-any-return]


class ConvOutputHead(nn.Module):
    """Conv output head decodes tokens to spatial output"""

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        patch_size: int,
        mid_channels: int = 64,
    ) -> None:
        super().__init__()
        if patch_size < 2 or (patch_size & (patch_size - 1)) != 0:
            raise ValueError(f"ConvOutputHead requires patch_size to be a power of 2 >= 2, got {patch_size}")
        self.patch_size = patch_size
        self.out_channels = out_channels

        factors = self._factorize(patch_size)
        self.stages = nn.ModuleList()

        for i, factor in enumerate(factors):
            is_first = i == 0
            is_last = i == len(factors) - 1
            ch_in = hidden_dim if is_first else mid_channels
            ch_out = out_channels if is_last else mid_channels

            layers: list[nn.Module] = [
                nn.Conv2d(ch_in, ch_in, 3, padding=1),
                nn.SiLU(),
            ]
            if is_first and len(factors) > 1:
                layers += [nn.Conv2d(ch_in, ch_in, 3, padding=1), nn.SiLU()]
            layers += [
                nn.Conv2d(ch_in, ch_out * factor**2, 1),
                nn.PixelShuffle(factor),
            ]
            self.stages.append(nn.Sequential(*layers))

    @staticmethod
    def _factorize(patch_size: int) -> list[int]:
        factors: list[int] = []
        remaining = patch_size
        while remaining % 4 == 0:
            factors.append(4)
            remaining //= 4
        if remaining == 2:
            factors.append(2)
        return factors

    def forward(
        self,
        x: Tensor,
        grid_h: int,
        grid_w: int,
    ) -> Tensor:
        """Decode tokens to spatial output via cascaded PixelShuffle stages.

        Args:
            x: Flattened tokens of shape ``(B, grid_h * grid_w, hidden_dim)``.
            grid_h: Patch grid height (``H // patch_size``).
            grid_w: Patch grid width (``W // patch_size``).

        Returns:
            Spatial tensor of shape ``(B, H, W, out_channels)`` after upsampling.
        """
        b = x.shape[0]
        x = x.view(b, grid_h, grid_w, -1)
        x = rearrange(x, "b h w c -> b c h w")
        for stage in self.stages:
            x = stage(x)
        return rearrange(x, "b c h w -> b h w c")
