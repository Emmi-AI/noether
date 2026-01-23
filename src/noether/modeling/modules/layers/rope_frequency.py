#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import warnings

import einops
import torch
from torch import nn

from noether.core.schemas.modules.layers import RopeFrequencyConfig
from noether.core.utils.torch import amp


class RopeFrequency(nn.Module):
    """Creates frequencies for rotary embeddings (RoPE) from https://arxiv.org/abs/2104.09864 for variable positions."""

    omega: torch.Tensor

    def __init__(
        self,
        config: RopeFrequencyConfig,
    ):
        """
        Args:
            config: Configuration for RoPE frequency settings.
        """
        super().__init__()
        if config.implementation == "real":
            warnings.warn(
                f"{self.__class__.__name__} implementation changed from real to complex because it is always "
                "faster, up to 70% for small models with low batch sizes. Consider setting implementation='complex' "
                "for improved speed."
            )
        self.hidden_dim = config.hidden_dim
        self.input_dim = config.input_dim
        self.implementation = config.implementation
        # if hidden_dim is not cleanly divisible -> cut away trailing dimensions
        self.ndim_padding = config.hidden_dim % config.input_dim
        dim_per_ndim = (config.hidden_dim - self.ndim_padding) // config.input_dim
        self.sincos_padding = dim_per_ndim % 2
        self.max_wavelength = config.max_wavelength
        self.padding = self.ndim_padding + self.sincos_padding * config.input_dim
        effective_dim_per_wave = (self.hidden_dim - self.padding) // config.input_dim
        assert effective_dim_per_wave > 0
        arange = torch.arange(0, effective_dim_per_wave, 2, dtype=torch.float)
        self.register_buffer(
            "omega",
            1.0 / config.max_wavelength ** (arange / effective_dim_per_wave),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        with amp.disable(device_type=str(coords.device).split(":")[0]):
            coordinate_ndim = coords.shape[-1]
            assert self.input_dim == coordinate_ndim
            out = coords.float().unsqueeze(-1) @ self.omega.unsqueeze(0)
        if self.implementation == "complex":
            out = einops.rearrange(out, "... input_dim hidden_dim -> ... (input_dim hidden_dim)")
            # add padding
            assert self.padding % 2 == 0
            out = torch.concat([out, torch.zeros(*out.shape[:-1], self.padding // 2, device=coords.device)], dim=-1)
            return torch.polar(torch.ones_like(out), out)
        # LEGACY: only kept for backward compatibility
        assert self.implementation == "real"
        if coords.ndim == 3:
            out = einops.repeat(
                out, "bs num_points input_dim hidden_dim -> bs num_points input_dim (two hidden_dim)", two=2
            )
        elif coords.ndim == 2:
            out = einops.repeat(out, "num_points input_dim hidden_dim -> num_points input_dim (two hidden_dim)", two=2)
        else:
            raise NotImplementedError
        return out.unbind(-2)

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(hidden_dim={self.hidden_dim})"
