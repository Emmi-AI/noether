#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import math
from typing import Literal, Self

import einops
import torch
from pydantic import BaseModel, Field, model_validator
from torch import nn

from noether.core.utils.torch import amp


class ContinuousSincosEmbeddingConfig(BaseModel):
    """Configuration for Continuous Sine-Cosine Embedding layer."""

    hidden_dim: int = Field(...)
    """Dimensionality of the output embedding."""
    input_dim: int = Field(...)
    """Dimensionality of the input coordinates."""
    mode: Literal["wavelength", "nerf"] = Field("wavelength")
    """Frequency schedule.

    - ``"wavelength"`` (default): transformer-style geometric wavelengths from ``1`` to
      ``max_wavelength``. Suitable for integer / unnormalized coordinates.
    - ``"nerf"``: NeRF-style log-spaced frequencies from ``π`` to ``π * max_frequency``.
      Suitable for coordinates normalized to ``[-1, 1]``. The ``L`` available bands
      are distributed evenly in log-frequency across this range.
    """
    max_wavelength: int = Field(10000)
    """Maximum wavelength. Only used when ``mode == "wavelength"``."""
    max_frequency: float | None = Field(None)
    """Highest frequency band for NeRF mode, in units of ``π``. The ``L`` frequencies
    are log-spaced between ``π`` (wavelength 2, spans the ``[-1, 1]`` domain) and
    ``π * max_frequency`` (wavelength ``2 / max_frequency``). Required when
    ``mode == "nerf"``; pick based on the smallest spatial scale you need to resolve
    in normalized coordinates (rough heuristic: ``1 / typical_point_spacing``)."""

    @model_validator(mode="after")
    def _check_mode_specific_fields(self) -> Self:
        if self.mode == "nerf" and self.max_frequency is None:
            raise ValueError("max_frequency is required when mode == 'nerf'")
        return self


class ContinuousSincosEmbed(nn.Module):
    """Embedding layer for continuous coordinates using sine and cosine functions.
    The original implementation from the Attenion is All You Need paper, deals with descrete 1D cordinates (i.e., a sequence).
    Howerver, this implementation is able to deal with 2D and 3D coordinate systems as well.

    Two frequency schedules are supported via ``config.mode``:

    - ``"wavelength"`` (default): geometric wavelengths from ``1`` to
      ``max_wavelength``, matching the original Transformer encoding. Use this for
      integer / unnormalized coordinates.
    - ``"nerf"``: log-spaced frequencies from ``π`` to ``π * max_frequency``. Use this
      for coordinates normalized to ``[-1, 1]``.
    """

    omega: torch.Tensor
    padding_tensor: torch.Tensor

    def __init__(
        self,
        config: ContinuousSincosEmbeddingConfig,
    ):
        """

        Args:
            config: Configuration for the ContinuousSincosEmbed module. See :class:`~noether.core.schemas.modules.layers.ContinuousSincosEmbeddingConfig` for the available options.
        """
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.input_dim = config.input_dim
        # if dim is not cleanly divisible -> cut away trailing dimensions
        self.ndim_padding = config.hidden_dim % config.input_dim
        dim_per_ndim = (config.hidden_dim - self.ndim_padding) // config.input_dim
        self.sincos_padding = dim_per_ndim % 2
        self.mode = config.mode
        self.max_wavelength = config.max_wavelength
        self.max_frequency = config.max_frequency
        self.padding = self.ndim_padding + self.sincos_padding * config.input_dim
        effective_dim_per_wave = (self.hidden_dim - self.padding) // config.input_dim
        assert effective_dim_per_wave > 0
        arange = torch.arange(0, effective_dim_per_wave, 2, dtype=torch.float32)
        if config.mode == "wavelength":
            omega = 1.0 / config.max_wavelength ** (arange / effective_dim_per_wave)
        else:  # "nerf"
            # L log-spaced frequencies between π and π * max_frequency.
            # Non-None enforced by the config's model_validator.
            assert config.max_frequency is not None
            num_bands = arange.numel()
            log2_freqs = torch.linspace(0.0, math.log2(config.max_frequency), num_bands)
            omega = math.pi * torch.pow(2.0, log2_freqs)
        self.register_buffer("omega", omega)
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
