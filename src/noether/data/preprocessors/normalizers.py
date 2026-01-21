#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

import torch

from noether.core.schemas.normalizers import (
    MeanStdNormalizerConfig,
    PositionNormalizerConfig,
    ShiftAndScaleNormalizerConfig,
)
from noether.data.preprocessors import PreProcessor, to_tensor
from noether.modeling.functional.logscale import from_logscale, to_logscale


class ShiftAndScaleNormalizer(PreProcessor):
    """Preprocessor that shifts and scales the input data, with (x + shift) * scale."""

    shift: torch.Tensor | None = None
    scale: torch.Tensor | None = None

    def __init__(
        self,
        normalizer_config: ShiftAndScaleNormalizerConfig,
        **kwargs,
    ):
        """

        Args:
            normalizer_config: Configuration containing shift and scale values.
        Raises:
            ValueError: If `shift` and `scale` do not have the same length.
            ValueError: If `logscale_shift` and `logscale_scale` do not have the same length when `logscale` is True.
            TypeError: If `shift`, `scale`, `logscale_shift`, or `logscale_scale` are not of type Sequence or torch.Tensor.
            ValueError: If `scale` contains zero values (to avoid division by zero).
            ValueError: If `scale` contains negative values.
            ValueError: If `shift` and `scale` are provided but not both.
        """
        super().__init__(**kwargs)

        self.scale = normalizer_config.scale
        self.shift = normalizer_config.shift
        self.logscale = normalizer_config.logscale

    def __call__(self, x: Any) -> Any:
        """Applies the shift and scale normalization to the input tensor.

        Args:
            x: torch.Tensor: The input tensor to normalize.

        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if self.logscale:
            x = to_logscale(x)
        return (x + self.shift) * self.scale

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalizes the input data by applying the inverse operation of the normalization.

        Args:
            x: torch.Tensor: The input tensor to denormalize.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        x = x * (1.0 / self.scale) - self.shift  # type: ignore[operator]
        if self.logscale:
            x = from_logscale(x)
        return x

    def __repr__(self) -> str:
        return f"ShiftAndScaleNormalizer(shift={self.shift}, scale={self.scale}, logscale={self.logscale})"


class MeanStdNormalization(ShiftAndScaleNormalizer):
    """Normalizes data using mean and standard deviation. It shifts the data by subtracting the mean and scales it by dividing by the standard deviation."""

    EPSILON = 1e-6  # Small value to avoid division by zero

    def __init__(self, normalizer_config: MeanStdNormalizerConfig, **kwargs):
        """

        Args:
            normalizer_config: Configuration containing mean and std values. Imports MeanStdNormalizerConfig.
            **kwargs: Additional arguments passed to the parent class.

        Raises:
            ValueError: If `mean` and `std` do not have the same length.
            ValueError: If any value in `std` is zero (to avoid division by zero).
            ValueError: If any value in `std` is negative.
        """

        self.mean = normalizer_config.mean
        self.std = normalizer_config.std  # Adding a small value to avoid division by zero

        if self.std.shape != self.mean.shape:
            raise ValueError("mean and std must have the same shape.")

        if (self.std == 0).any():
            raise ValueError("std must not contain zero values to avoid division by zero.")

        if (self.std < 0).any():
            raise ValueError("std must not contain negative values.")

        shift = -self.mean
        scale = torch.reciprocal(self.std.clamp(min=self.EPSILON))
        config = ShiftAndScaleNormalizerConfig(shift=shift, scale=scale, logscale=normalizer_config.logscale)
        super().__init__(normalizer_config=config, **kwargs)


class PositionNormalizer(ShiftAndScaleNormalizer):
    """Normalizes position data to a range of [0, scale]. It inherits from ShiftAndScaleNormalizer and applies a shift and scale based on the provided raw position min and max values."""

    def __init__(
        self,
        normalizer_config: PositionNormalizerConfig,
        **kwargs,
    ):
        """

        Args:
            normalizer_config: Configuration containing raw position min, max, and scale values.
            **kwargs: Additional arguments passed to the parent class.

        Raises:
            ValueError: If `raw_pos_min` and `raw_pos_max` do not have the same length.
            ValueError: If `raw_pos_max` is equal to `raw_pos_min`.
            ValueError: If `scale` is not a positive number.
        """

        self.raw_pos_min = normalizer_config.raw_pos_min
        self.raw_pos_max = normalizer_config.raw_pos_max
        # Do not remove this. The scale variable is not the same as we pass to the ShiftAndScaleNormalizer.
        # It is used to scale the coordinates to a range of [0, scale]. However, we need to recompute the scale based on the raw position min and max values.
        scale = to_tensor(normalizer_config.scale)

        self.resizing_scale = scale  # this is a reference to the input scale, not the computed scale

        scale = scale / (self.raw_pos_max - self.raw_pos_min)
        shift = -self.raw_pos_min

        super().__init__(
            normalizer_config=ShiftAndScaleNormalizerConfig(
                shift=shift,
                scale=scale,
            ),
            **kwargs,
        )

    def __call__(self, x: Any) -> Any:
        """Applies the position normalization to the input tensor.

        Args:
            x: torch.Tensor: The input tensor to normalize.

        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        output = super().__call__(x)  # type: ignore[return-value]
        if torch.any(output < 0) or torch.any(output > self.resizing_scale):
            raise ValueError("Normalized positions are out of bounds [0, scale].")

        return output
