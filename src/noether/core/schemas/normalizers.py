#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections.abc import Sequence
from typing import Annotated, Any, ClassVar, Literal, Self, Union

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, PlainValidator, model_validator

from noether.core.schemas.lib import _RegistryBase


# 1. Define a function to validate the input.
# It will accept a value and try to convert it to a torch.Tensor.
def validate_tensor(v: Any) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        return v
    # You can add more robust conversion logic here, e.g., from numpy
    if isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    try:
        # Assumes input is a list or similar structure
        return torch.tensor(v)
    except Exception as e:
        raise ValueError(f"Could not convert {v} to torch.Tensor: {e}") from None


# 2. Define the custom Tensor type using Annotated.
# This is the modern Pydantic V2 approach.
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


class MeanStdNormalizerConfig(NormalizerConfig):
    kind: str | None = "noether.data.preprocessors.normalizers.MeanStdNormalization"
    mean: TorchTensor
    """mean to subtract from the input data. Can be a single value or a Sequence if we want to apply a different mean per dimension."""
    std: TorchTensor
    """standard deviation to divide the input data by. Can be a single value or a Sequence if we want to apply a different std per dimension."""
    logscale: bool = False
    """If true, the input data is assumed to be in log scale."""


class PositionNormalizerConfig(NormalizerConfig):
    kind: str | None = "noether.data.preprocessors.normalizers.PositionNormalizer"
    raw_pos_min: TorchTensor
    """Minimum raw position values of the entire simulation mesh. Can be a single value or a sequence of values."""
    raw_pos_max: TorchTensor
    """Maximum raw position values of the entire simulation mesh. Can be a single value or a sequence of values."""
    scale: float = Field(default=1000.0, gt=0.0)
    """Scaling factor, the coordinates will be scaled linearly between [0, scale]. Defaults to 1000."""

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


class FieldNormalizerConfig(NormalizerConfig):
    """Declarative normalizer config that references dataset statistics by convention.

    Instead of embedding numeric values (mean, std, etc.) directly, this config declares
    *how* to normalize a field. The actual statistics are resolved at runtime from the
    dataset's statistics file.

    For ``"mean_std"`` normalization, the builder looks up ``{field}_mean`` and
    ``{field}_std`` in the dataset statistics (customizable via ``stat_keys``).

    For ``"position"`` normalization, the builder looks up ``raw_pos_min`` and
    ``raw_pos_max`` (customizable via ``stat_keys``).
    """

    kind: str | None = "noether.data.preprocessors.normalizers.FieldNormalizer"

    strategy: Literal["mean_std", "position"] = "mean_std"  # type: ignore[assignment]
    """Normalization strategy. ``"mean_std"`` for mean/std normalization, ``"position"`` for position normalization."""
    logscale: bool = False
    """If true, the input data is converted to log scale before normalization. Only used for ``"mean_std"``."""
    stat_keys: dict[str, str] | None = None
    """Optional overrides for statistic key lookup. For ``"mean_std"``: ``{"mean": "custom_mean_key", "std": "custom_std_key"}``.
    For ``"position"``: ``{"min": "custom_min_key", "max": "custom_max_key"}``."""
    scale: float = Field(default=1000.0, gt=0.0)
    """Scaling factor for position normalization. Coordinates are scaled to [0, scale]. Only used for ``"position"``."""


AnyNormalizer = Union[
    MeanStdNormalizerConfig, PositionNormalizerConfig, ShiftAndScaleNormalizerConfig, FieldNormalizerConfig
]
