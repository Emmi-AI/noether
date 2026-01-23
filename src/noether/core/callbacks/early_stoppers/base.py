#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import abc

from noether.core.callbacks.periodic import IntervalType, PeriodicCallback
from noether.core.utils.training import UpdateCounter


class EarlyStopIteration(StopIteration):
    """Custom StopIteration exception for Early Stoppers."""


class EarlyStopperBase(PeriodicCallback, metaclass=abc.ABCMeta):
    """Base class for early stoppers that is used to define the interface for early stoppers used by the trainers."""

    def to_short_interval_string(self) -> str:
        """Convert the interval to a short string representation used for logging."""
        intervals = [
            (self.every_n_epochs, "E"),
            (self.every_n_updates, "U"),
            (self.every_n_samples, "S"),
        ]
        return "_".join(f"{prefix}{val}" for val, prefix in intervals if val is not None)

    def periodic_callback(
        self,
        *,
        interval_type: IntervalType,
        update_counter: UpdateCounter,
        **kwargs,
    ) -> None:
        if interval_type == "eval":
            return  # early stopping is only applied during training
        if self._should_stop(update_counter=update_counter):
            raise EarlyStopIteration

    @abc.abstractmethod
    def _should_stop(self, *, update_counter: UpdateCounter):
        raise NotImplementedError("This method should be implemented by the derived class.")
