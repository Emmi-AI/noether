#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


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
