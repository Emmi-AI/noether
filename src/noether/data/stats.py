#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import math
from dataclasses import dataclass
from typing import Union

import torch

from noether.modeling.functional.logscale import to_logscale


class RunningMoments(torch.nn.Module):
    """
    Calculates running moments of data (mean, variance, min, max) for data normalization purposes.

    This class implements Welford's online algorithm for computing mean and variance
    in a single pass, making it memory-efficient for large datasets.
    """

    _m: torch.Tensor
    _s: torch.Tensor | None
    _min: torch.Tensor | None
    _max: torch.Tensor | None
    _n: int

    def __init__(self, log_scale: bool = False):
        """
        Initialize a RunningMoments instance.

        Args:
            log_scale: Whether to apply log scaling to the data before calculating moments.
                       This affects all operations unless overridden.
        """
        super().__init__()
        self.log_scale = log_scale
        self._n = 0
        # register_buffer expects a Tensor, but we init to 0 scalar for _m
        self.register_buffer("_m", torch.tensor(0.0, dtype=torch.float64))
        # Initialize optional buffers as attributes; they will be registered/assigned later
        self.register_buffer("_s", None)
        self.register_buffer("_min", None)
        self.register_buffer("_max", None)

    def push_tensor(self, x: torch.Tensor, dim: int = 1, log_scale: bool | None = None) -> None:
        """
        Add a tensor to the moment calculation. Calculations are carried out in float64 to avoid numerical
        imprecisions.

        Args:
            x: Tensor to push. Shape should be compatible with dim parameter.
            dim: Which dimension contains the feature dimension. For example:
                 - In a pointcloud (N, 3), dim=1 calculates statistics per coordinate (x, y, z)
                 - In an RGB image (B, C, H, W), dim=1 calculates per-channel statistics
            log_scale: Whether to apply log scaling to this tensor. If None, uses the class default.

        Raises:
            ValueError: If tensor has invalid dimensions or is incompatible with previous data
        """
        if not torch.is_tensor(x):
            raise TypeError(f"Expected a torch.Tensor, got {type(x)}")

        # Determine whether to use log scale
        use_log_scale = self.log_scale if log_scale is None else log_scale

        # Convert to float64 for numerical stability
        x = x.to(self._m.device, dtype=torch.float64, non_blocking=True)  # type: ignore[has-type]
        if use_log_scale:
            x = to_logscale(x)

        if x.ndim < 1:
            raise ValueError("Input tensor must have at least one dimension")

        if x.ndim == 1:
            # Only 1 feature (e.g., pressure) -> unsqueeze feature dim
            if dim != 1:
                raise ValueError(f"For 1D tensors, dim must be 1, got {dim}")
            x = x.unsqueeze(1)

        # Move feature dim to 0
        if dim != 0:
            x = x.transpose(0, dim)

        # Flatten all non-feature dims
        x = x.flatten(start_dim=1)

        batch_n = x.size(1)
        batch_mean = x.mean(dim=1)
        batch_sq_diff = ((x - batch_mean.unsqueeze(1)) ** 2).sum(dim=1)
        current_min, current_max = torch.aminmax(x, dim=1)

        if self._n == 0:
            self._n = batch_n
            self._m = batch_mean
            self._s = batch_sq_diff
            self._min = current_min
            self._max = current_max
        else:
            # Type guard: if n > 0, these must be tensors
            if self._s is None or self._min is None or self._max is None:
                raise RuntimeError("Internal state error: Stats are None but _n > 0")

            if batch_mean.shape != self._m.shape:
                raise ValueError(
                    f"Incompatible tensor shape: expected features of shape {self._m.shape}, got {batch_mean.shape}"
                )

            delta = batch_mean - self._m
            new_n = self._n + batch_n

            self._m += delta * batch_n / new_n
            self._s += batch_sq_diff + delta**2 * self._n * batch_n / new_n
            self._n += batch_n

            # Use maximum/minimum for element-wise operations with 'out'
            torch.maximum(self._max, current_max, out=self._max)
            torch.minimum(self._min, current_min, out=self._min)

    def push_scalar(self, x: float, log_scale: bool | None = None) -> None:
        """
        Add a scalar to the moment calculation. Calculations are carried out in float64.

        Args:
            x: Scalar value to add to the moment calculations.
            log_scale: Whether to apply log scaling to this scalar. If None, uses the class default.

        Raises:
            TypeError: If x is not a Python float
        """
        if not isinstance(x, float):
            raise TypeError("Expected a Python float (this enforces float64 precision)")

        self.push_tensor(torch.tensor([x], dtype=torch.float64, device=self._m.device), dim=1, log_scale=log_scale)

    @property
    def mean(self) -> Union[float, torch.Tensor]:
        if self._n <= 0:
            raise ValueError("No data has been pushed yet")
        return self._m

    @property
    def var(self) -> Union[float, torch.Tensor]:
        if self._n <= 1 or self._s is None:
            raise ValueError("Need at least 2 samples to calculate variance")
        return self._s / (self._n - 1)

    @property
    def std(self) -> Union[float, torch.Tensor]:
        var = self.var
        if isinstance(var, torch.Tensor):
            return torch.sqrt(var)
        return math.sqrt(var)

    @property
    def min(self) -> Union[float, torch.Tensor]:
        if self._min is None:
            raise ValueError("No data has been pushed yet")
        return self._min.item() if self._min.numel() == 1 else self._min

    @property
    def max(self) -> Union[float, torch.Tensor]:
        if self._max is None:
            raise ValueError("No data has been pushed yet")
        return self._max.item() if self._max.numel() == 1 else self._max

    @property
    def count(self) -> int:
        return self._n

    def print(self) -> None:
        """Print the calculated statistics in a readable format."""

        def format_value(val: float | torch.Tensor) -> str:
            """Format a value (scalar or tensor) as a clean string."""
            if isinstance(val, torch.Tensor):
                if val.numel() == 1:
                    return f"({val.item():.5e},)"
                else:
                    # Format each element with scientific notation
                    values = [f"{v:.5e}" for v in val.tolist()]
                    return f"({', '.join(values)})"
            else:
                return f"{val:.5e}"

        print(f"  Mean: {format_value(self.mean)}")
        print(f"  Std:  {format_value(self.std)}")
        print(f"  Var:  {format_value(self.var)}")
        print(f"  Min:  {format_value(self.min)}")
        print(f"  Max:  {format_value(self.max)}")
        print(f"  Count: {self.count}")

    def dump(self) -> dict[str, Union[float, torch.Tensor]]:
        return {
            "mean": self.mean,
            "std": self.std,
            "var": self.var,
            "min": self.min,
            "max": self.max,
            "count": self.count,
        }

    def reset(self) -> None:
        """Reset the statistics, clearing all accumulated data."""
        device = self._m.device
        self._n = 0
        self._m = torch.tensor(0.0, dtype=torch.float64, device=device)
        self._s = None
        self._min = None
        self._max = None

    def is_empty(self) -> bool:
        return self._n == 0


@dataclass
class _Stats:
    min: torch.Tensor
    max: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    logmean: torch.Tensor
    logstd: torch.Tensor


class RunningStats:
    """Calculates statistics of data (min, max, mean and variance) for data normalization purposes."""

    def __init__(self, name: str | None = None):
        self.name = name
        self._n = 0
        self._stats: _Stats | None = None

    def push_tensor(self, x: torch.Tensor, dim: int = 1) -> None:
        """Add a tensor to the statistics. Calculations are carried out in float64 to avoid numerical imprecision."""
        assert x.ndim >= 1
        x = x.double()
        if x.ndim == 1:
            # only 1 feature (e.g., pressure) -> unsqueeze feature dim
            assert dim == 1
            x = x.unsqueeze(1)
        # move feature dim to 0
        if dim != 0:
            x = x.transpose(0, dim)
        # flatten all non-feature dims
        x = x.flatten(start_dim=1)
        logx = torch.sign(x) * torch.log1p(x.abs())

        batch_n = x.size(1)
        # Explicitly access values tuple from min/max for type clarity
        batch_min = x.min(dim=1).values
        batch_max = x.max(dim=1).values
        batch_mean = x.mean(dim=1)
        batch_std = ((x - batch_mean.unsqueeze(1)) ** 2).sum(dim=1)
        batch_logmean = logx.mean(dim=1)
        batch_logstd = ((logx - batch_logmean.unsqueeze(1)) ** 2).sum(dim=1)

        if self._n == 0:
            self._n = batch_n
            self._stats = _Stats(
                min=batch_min,
                max=batch_max,
                mean=batch_mean,
                std=batch_std,
                logmean=batch_logmean,
                logstd=batch_logstd,
            )
        else:
            assert self._stats is not None, "Stats should not be None when _n > 0"
            assert batch_min.shape == self._stats.min.shape
            assert batch_max.shape == self._stats.max.shape
            assert batch_mean.shape == self._stats.mean.shape
            assert batch_std.shape == self._stats.std.shape

            delta = batch_mean - self._stats.mean
            logdelta = batch_logmean - self._stats.logmean
            total_n = self._n + batch_n

            self._stats = _Stats(
                min=torch.minimum(self._stats.min, batch_min),
                max=torch.maximum(self._stats.max, batch_max),
                mean=self._stats.mean + delta * batch_n / total_n,
                std=self._stats.std + batch_std + delta**2 * self._n * batch_n / total_n,
                logmean=self._stats.logmean + logdelta * batch_n / total_n,
                logstd=self._stats.logstd + batch_logstd + logdelta**2 * self._n * batch_n / total_n,
            )
            self._n = total_n

    @property
    def min(self) -> torch.Tensor:
        assert self._n > 0 and self._stats is not None
        return self._stats.min

    @property
    def max(self) -> torch.Tensor:
        assert self._n > 0 and self._stats is not None
        return self._stats.max

    @property
    def mean(self) -> torch.Tensor:
        assert self._n > 0 and self._stats is not None
        return self._stats.mean

    @property
    def var(self) -> torch.Tensor:
        assert self._n > 1 and self._stats is not None
        return self._stats.std / (self._n - 1)

    @property
    def std(self) -> torch.Tensor:
        return self.var.pow(0.5)

    @property
    def logmean(self) -> torch.Tensor:
        assert self._n > 0 and self._stats is not None
        return self._stats.logmean

    @property
    def logvar(self) -> torch.Tensor:
        assert self._n > 1 and self._stats is not None
        return self._stats.logstd / (self._n - 1)

    @property
    def logstd(self) -> torch.Tensor:
        return self.logvar.pow(0.5)

    def __str__(self) -> str:
        return "\n".join(
            [
                "-" * 50,
                *([] if self.name is None else [self.name]),
                f"min: {self.min}",
                f"max: {self.max}",
                f"mean: {self.mean}",
                f"std: {self.std}",
                f"logmean: {self.logmean}",
                f"logstd: {self.logstd}",
                "-" * 50,
            ],
        )
