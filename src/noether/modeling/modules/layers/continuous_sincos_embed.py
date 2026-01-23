#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import einops
import torch
from torch import nn

from noether.core.schemas.modules.layers import ContinuousSincosEmbeddingConfig
from noether.core.utils.torch import amp


class ContinuousSincosEmbed(nn.Module):
    """Embedding layer for continuous coordinates using sine and cosine functions.
    The original implementation from the Attenion is All You Need paper, deals with descrete 1D cordinates (i.e., a sequence).
    Howerver, this implementation is able to deal with 2D and 3D coordinate systems as well.

    """

    omega: torch.Tensor
    padding_tensor: torch.Tensor

    def __init__(
        self,
        config: ContinuousSincosEmbeddingConfig,
    ):
        """Initialize the ContinuousSincosEmbed layer.

        Args:
            config: Instance of `ContinuousSincosEmbeddingConfig`.
        """
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.input_dim = config.input_dim
        # if dim is not cleanly divisible -> cut away trailing dimensions
        self.ndim_padding = config.hidden_dim % config.input_dim
        dim_per_ndim = (config.hidden_dim - self.ndim_padding) // config.input_dim
        self.sincos_padding = dim_per_ndim % 2
        self.max_wavelength = config.max_wavelength
        self.padding = self.ndim_padding + self.sincos_padding * config.input_dim
        effective_dim_per_wave = (self.hidden_dim - self.padding) // config.input_dim
        assert effective_dim_per_wave > 0
        arange = torch.arange(0, effective_dim_per_wave, 2, dtype=torch.float32)
        self.register_buffer(
            "omega",
            1.0 / config.max_wavelength ** (arange / effective_dim_per_wave),
        )
        self.register_buffer("padding_tensor", torch.zeros(self.padding, dtype=torch.float32))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward method of the ContinuousSincosEmbed layer.

        Args:
            coords: Tensor of coordinates. The shape of the tensor should be [batch size, number of points, coordinate dimension] or [number of points, coordinate dimension].

        Raises:
            NotImplementedError: Only supports sparse (i.e. [number of points, coordinate dimension]) or dense (i.e. [batch size, number of points, coordinate dimension]) coordinates systems.

        Returns:
            Tensor with embedded coordinates.
        """

        # fp32 to avoid numerical imprecision
        coords = coords.float()
        with amp.disable(device_type=str(coords.device).split(":")[0]):
            coordinate_ndim = coords.shape[-1]
            assert self.input_dim == coordinate_ndim
            out = coords.unsqueeze(-1) @ self.omega.unsqueeze(0)
            emb = torch.concat([torch.sin(out), torch.cos(out)], dim=-1)
            if coords.ndim == 3:
                emb = einops.rearrange(
                    emb, "bs num_points input_dim hidden_dim -> bs num_points (input_dim hidden_dim)"
                )
            elif coords.ndim == 2:
                emb = einops.rearrange(emb, "num_points input_dim hidden_dim -> num_points (input_dim hidden_dim)")
            else:
                raise NotImplementedError
        if self.padding > 0:
            padding = self.padding_tensor.expand(*emb.shape[:-1], -1)
            emb = torch.concat([emb, padding], dim=-1)
        return emb

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dim={self.hidden_dim})"
