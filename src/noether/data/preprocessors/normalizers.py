#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections.abc import Sequence
from typing import Annotated, Any, ClassVar, Literal, Self, Union

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, PlainValidator, model_validator

from noether.core.schemas.lib import _RegistryBase
from noether.data.preprocessors import PreProcessor, to_tensor
from noether.modeling.functional.logscale import from_logscale, to_logscale


def validate_tensor(v: Any) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        return v
    if isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    try:
        return torch.tensor(v)
    except Exception as e:
        raise ValueError(f"Could not convert {v} to torch.Tensor: {e}") from None


TorchTensor = Annotated[
    torch.Tensor,
    PlainValidator(validate_tensor),
    PlainSerializer(lambda x: x.tolist(), return_type=list, when_used="always"),
]

FloatOrArray = float | Sequence[float] | TorchTensor
SequenceOrTensor = Sequence[float] | TorchTensor


class NormalizerConfig(_RegistryBase):
    """Base configuration for normalizers. All normalizer configs should inherit from this class."""

    _registry: ClassVar[dict[str, type[BaseModel]]] = {}
    _type_field: ClassVar[str] = "kind"
    kind: str | None = None
    """Kind of normalizer to use, i.e. class path"""

    model_config = ConfigDict(extra="forbid")


class ShiftAndScaleNormalizerConfig(NormalizerConfig):
    kind: str | None = "noether.data.preprocessors.normalizers.ShiftAndScaleNormalizer"
    shift: TorchTensor
    """Value to subtract from the input data. Can be a single value or a Sequence if we want to apply a different shift per dimension.
    Assumed in log scale if logscale is True.
    """
    scale: TorchTensor
    """Value to divide the input data by. Can be a single value or a Sequence if we want to apply a different scale per dimension.
    Assumed in log scale if logscale is True.
    """
    logscale: bool = False
    """If true, the input data is assumed to be in log scale."""

    @model_validator(mode="after")
    def check_shift_scale(self) -> Self:
        if self.shift.shape != self.scale.shape:
            raise ValueError("shift and scale must have the same shape.")

        comp = self.scale <= 0.0
        if torch.any(comp):
            raise ValueError(
                f"scale must be a positive number. Erroneous indices: {torch.nonzero(comp).squeeze().tolist()}"
            )

        return self


class ShiftAndScaleNormalizer(PreProcessor):
    """Preprocessor that shifts and scales the input data, with (x + shift) * scale."""

    def __init__(
        self,
        normalizer_config: ShiftAndScaleNormalizerConfig,
        **kwargs,
    ):
        """

        Args:
            normalizer_config: Configuration containing shift and scale values. See :class:`~noether.core.schemas.normalizers.ShiftAndScaleNormalizerConfig` for details.
            **kwargs: Additional arguments passed to the parent class.

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

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the shift and scale normalization to the input tensor.

        Args:
            x: torch.Tensor: The input tensor to normalize.

        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if self.logscale:
            x = to_logscale(x)
        return (x + self.shift.to(x.device)) * self.scale.to(x.device)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalizes the input data by applying the inverse operation of the normalization.

        Args:
            x: torch.Tensor: The input tensor to denormalize.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        x = x * (1.0 / self.scale.to(x.device)) - self.shift.to(x.device)  # type: ignore[operator]
        if self.logscale:
            x = from_logscale(x)
        return x

    def __repr__(self) -> str:
        return f"ShiftAndScaleNormalizer(shift={self.shift}, scale={self.scale}, logscale={self.logscale})"


class MeanStdNormalizerConfig(NormalizerConfig):
    kind: str | None = "noether.data.preprocessors.normalizers.MeanStdNormalization"
    mean: TorchTensor
    """mean to subtract from the input data. Can be a single value or a Sequence if we want to apply a different mean per dimension."""
    std: TorchTensor
    """standard deviation to divide the input data by. Can be a single value or a Sequence if we want to apply a different std per dimension."""
    logscale: bool = False
    """If true, the input data is assumed to be in log scale."""


class MeanStdNormalization(ShiftAndScaleNormalizer):
    """Normalizes data using mean and standard deviation. It shifts the data by subtracting the mean and scales it by dividing by the standard deviation."""

    EPSILON = 1e-6  # Small value to avoid division by zero

    def __init__(self, normalizer_config: MeanStdNormalizerConfig, **kwargs):
        """

        Args:
            normalizer_config: Configuration containing mean and std values. See :class:`~noether.core.schemas.normalizers.MeanStdNormalizerConfig` for details.
            **kwargs: Additional arguments passed to the parent class.

        Raises:
            ValueError: If `mean` and `std` do not have the same length.
            ValueError: If any value in `std` is zero (to avoid division by zero).
            ValueError: If any value in `std` is negative.
        """

        self.mean = normalizer_config.mean
        self.std = normalizer_config.std

        if self.std.shape != self.mean.shape:
            raise ValueError("mean and std must have the same shape.")

        if (self.std == 0).any():
            raise ValueError("std must not contain zero values to avoid division by zero.")

        if (self.std < 0).any():
            raise ValueError("std must not contain negative values.")

        shift = -self.mean
        scale = torch.reciprocal(self.std.clamp(min=self.EPSILON))  # Adding a small value to avoid division by zero
        config = ShiftAndScaleNormalizerConfig(shift=shift, scale=scale, logscale=normalizer_config.logscale)
        super().__init__(normalizer_config=config, **kwargs)


class PositionNormalizerConfig(NormalizerConfig):
    kind: str | None = "noether.data.preprocessors.normalizers.PositionNormalizer"
    raw_pos_min: TorchTensor
    """Minimum raw position values of the entire simulation mesh. Can be a single value or a sequence of values."""
    raw_pos_max: TorchTensor
    """Maximum raw position values of the entire simulation mesh. Can be a single value or a sequence of values."""
    scale: float = Field(default=1000.0, gt=0.0)
    """Scaling factor, the coordinates will be scaled linearly between [0, scale] (or [-scale, scale] if ``zero_center`` is True). Defaults to 1000."""
    zero_center: bool = False
    """If True, coordinates are scaled to [-scale, scale] instead of [0, scale]."""

    @model_validator(mode="after")
    def check_min_max(self) -> Self:
        if self.raw_pos_max.shape != self.raw_pos_min.shape:
            raise ValueError("raw_pos_min and raw_pos_max must have the same shape.")

        comp = self.raw_pos_max <= self.raw_pos_min
        if torch.any(comp):
            raise ValueError(
                f"raw_pos_max must be element-wise greater than raw_pos_min. Errenous indices: {torch.nonzero(comp).squeeze().tolist()}"
            )

        return self


class PositionNormalizer(ShiftAndScaleNormalizer):
    """Normalizes position data to a range of [0, scale], or [-scale, scale] when ``zero_center`` is True. It inherits from ShiftAndScaleNormalizer and applies a shift and scale based on the provided raw position min and max values."""

    EPSILON = 1e-6  # Small value to check bounds with some tolerance

    def __init__(
        self,
        normalizer_config: PositionNormalizerConfig,
        **kwargs,
    ):
        """

        Args:
            normalizer_config: Configuration containing raw position min, max, scale, and zero_center values. See :class:`~noether.core.schemas.normalizers.PositionNormalizerConfig` for details.
            **kwargs: Additional arguments passed to the parent class.

        Raises:
            ValueError: If `raw_pos_min` and `raw_pos_max` do not have the same length.
            ValueError: If `raw_pos_max` is equal to `raw_pos_min`.
            ValueError: If `scale` is not a positive number.
        """

        self.raw_pos_min = normalizer_config.raw_pos_min
        self.raw_pos_max = normalizer_config.raw_pos_max
        self.zero_center = normalizer_config.zero_center
        # Do not remove this. The scale variable is not the same as we pass to the ShiftAndScaleNormalizer.
        # It is used to scale the coordinates to a range of [0, scale] (or [-scale, scale] if zero_center is True).
        # However, we need to recompute the scale based on the raw position min and max values.
        scale = to_tensor(normalizer_config.scale)

        self.resizing_scale = scale  # this is a reference to the input scale, not the computed scale

        if self.zero_center:
            shift = -(self.raw_pos_max + self.raw_pos_min) / 2
            scale = 2 * scale / (self.raw_pos_max - self.raw_pos_min)
        else:
            shift = -self.raw_pos_min
            scale = scale / (self.raw_pos_max - self.raw_pos_min)

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
        upper = self.resizing_scale.to(x.device)
        lower = -upper if self.zero_center else torch.zeros_like(upper)
        if torch.any(output < lower - self.EPSILON) or torch.any(output > upper + self.EPSILON):
            bounds = "[-scale, scale]" if self.zero_center else "[0, scale]"
            raise ValueError(f"Normalized values are out of bounds {bounds}.")

        return output


class FieldNormalizerConfig(NormalizerConfig):
    """Declarative normalizer config that references dataset statistics by convention.

    Instead of embedding numeric values (mean, std, etc.) directly, this config declares
    *how* to normalize a field. The actual statistics are resolved at runtime from the
    dataset's statistics file.

    For ``"mean_std"`` normalization, the builder looks up ``{field}_mean`` and
    ``{field}_std`` in the dataset statistics (customizable via ``stat_keys``).

    For ``"min_max"`` normalization, the builder looks up ``{field}_min`` and
    ``{field}_max`` (customizable via ``stat_keys``).
    """

    kind: str | None = "noether.data.preprocessors.normalizers.FieldNormalizer"

    strategy: Literal["mean_std", "position", "min_max"] = "mean_std"  # type: ignore[assignment]
    """Normalization strategy. ``"mean_std"`` for mean/std normalization, ``"min_max"`` for min/max normalization.``"position"`` is an alias for min_max, """
    logscale: bool = False
    """If true, the input data is converted to log scale before normalization. Only used for ``"mean_std"``."""
    stat_keys: dict[str, str] | None = None
    """Optional overrides for statistic key lookup. For ``"mean_std"``: ``{"mean": "custom_mean_key", "std": "custom_std_key"}``.
    For ``"min_max/position"``: ``{"min": "custom_min_key", "max": "custom_max_key"}``."""
    scale: float = Field(default=1000.0, gt=0.0)
    """Scaling factor for position normalization. Coordinates are scaled to [0, scale]. Only used for ``"position"``."""
    zero_center: bool = False
    """If True, position normalization is zero-centered (scaled to [-scale, scale]) instead of [0, scale]. Only used for ``"position"``."""


class FieldNormalizer(PreProcessor):
    """Preprocessor that normalizes a field based on a specified strategy and dataset statistics."""

    normalizer: PreProcessor

    def __init__(
        self,
        normalizer_config: FieldNormalizerConfig,
        statistics: dict[str, list[float | int] | float | int] | None,
        **kwargs,
    ):
        """

        Args:
            normalizer_config: Configuration containing the normalization strategy and logscale flag. See :class:`~noether.core.schemas.normalizers.FieldNormalizerConfig` for details.
            statistics: A dictionary containing the dataset statistics needed for normalization (e.g., mean, std, raw_pos_min, raw_pos_max).
            **kwargs: Additional arguments passed to the parent class.

        Raises:
            ValueError: If the required statistics for the chosen strategy are not present in the `statistics` dictionary.
            ValueError: If the normalization strategy is not supported.
        """
        super().__init__(**kwargs)

        stat_keys = normalizer_config.stat_keys or {}

        if statistics is None:
            raise ValueError("Statistics must be provided for FieldNormalizer.")

        if normalizer_config.strategy == "mean_std":
            mean_key = stat_keys.get("mean", f"{self.normalization_key}_mean")
            std_key = stat_keys.get("std", f"{self.normalization_key}_std")
            mean_val = statistics[mean_key]
            std_val = statistics[std_key]
            if isinstance(mean_val, (int, float)):
                mean_val = [mean_val]
            if isinstance(std_val, (int, float)):
                std_val = [std_val]
            self.normalizer = MeanStdNormalization(
                MeanStdNormalizerConfig(
                    mean=mean_val,
                    std=std_val,
                    logscale=normalizer_config.logscale,
                ),
                normalization_key=self.normalization_key,
            )
        elif normalizer_config.strategy == "position" or normalizer_config.strategy == "min_max":
            min_key = stat_keys.get("min", f"{self.normalization_key}_min")
            max_key = stat_keys.get("max", f"{self.normalization_key}_max")
            if min_key not in statistics:
                raise ValueError(
                    f"Missing required statistics for position normalization: '{min_key}' and/or '{max_key}' not found in statistics."
                )
            if max_key not in statistics:
                raise ValueError(
                    f"Missing required statistics for position normalization: '{min_key}' and/or '{max_key}' not found in statistics."
                )
            self.normalizer = PositionNormalizer(
                PositionNormalizerConfig(
                    raw_pos_min=statistics[min_key],
                    raw_pos_max=statistics[max_key],
                    scale=normalizer_config.scale,
                    zero_center=normalizer_config.zero_center,
                ),
                normalization_key=self.normalization_key,
            )
        else:
            raise ValueError(f"Unknown normalizer type '{normalizer_config.strategy}'")

    def __call__(self, x: Any) -> Any:
        return self.normalizer(x)  # type: ignore[return-value]

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalizer.denormalize(x)  # type: ignore[return-value]


AnyNormalizer = Union[
    MeanStdNormalizerConfig, PositionNormalizerConfig, ShiftAndScaleNormalizerConfig, FieldNormalizerConfig
]
