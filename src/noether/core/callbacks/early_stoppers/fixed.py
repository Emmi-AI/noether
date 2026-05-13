#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from noether.core.callbacks.base import CallBackBaseConfig
from noether.core.callbacks.early_stoppers.base import EarlyStopperBase
from noether.core.utils.training import UpdateCounter


class FixedEarlyStopperConfig(BaseModel):
    kind: str | None = None
    name: Literal["FixedEarlyStopper"] = Field("FixedEarlyStopper", frozen=True)
    stop_at_sample: int | None = None
    stop_at_update: int | None = None
    stop_at_epoch: int | None = None

    @model_validator(mode="after")
    def validate_callback_frequency(self) -> FixedEarlyStopperConfig:
        """
        Ensures that exactly one stop ('stop_at_*') is specified
        """
        # 1. Mutual Exclusivity and Presence Validation
        frequency_fields = [self.stop_at_epoch, self.stop_at_update, self.stop_at_sample]
        num_frequency_fields_set = sum(1 for f in frequency_fields if f is not None)

        if num_frequency_fields_set != 1:
            raise ValueError(
                "Exactly one of 'stop_at_epoch', 'stop_at_update', or 'stop_at_sample' must be set. Cannot have multiple or none set."
            )
        return self


class FixedEarlyStopper(EarlyStopperBase):
    """Early stopper (training) based on a fixed number of epochs, updates, or samples.

    Example config:

    .. code-block:: yaml

        - kind: noether.core.callbacks.FixedEarlyStopper
          stop_at_epoch: 10
          name: FixedEarlyStopper

    """

    def __init__(
        self,
        callback_config: FixedEarlyStopperConfig,
        **kwargs,
    ):
        """

        Args:
            callback_config: The configuration for the callback. See
                :class:`~noether.core.schemas.callbacks.FixedEarlyStopperConfig`
                for available options.
            **kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__(CallBackBaseConfig.model_validate(dict(every_n_updates=1)), **kwargs)
        self.stop_at_sample = callback_config.stop_at_sample
        self.stop_at_update = callback_config.stop_at_update
        self.stop_at_epoch = callback_config.stop_at_epoch
        if self.stop_at_sample is None and self.stop_at_update is None and self.stop_at_epoch is None:
            raise ValueError("at least one of stop_at_sample, stop_at_update, stop_at_epoch must be set")

    def _should_stop(self, *, update_counter: UpdateCounter):
        return (
            (
                self.stop_at_sample is not None
                and update_counter.cur_iteration.sample is not None
                and update_counter.cur_iteration.sample >= self.stop_at_sample
            )
            or (
                self.stop_at_update is not None
                and update_counter.cur_iteration.update is not None
                and update_counter.cur_iteration.update >= self.stop_at_update
            )
            or (
                self.stop_at_epoch is not None
                and update_counter.cur_iteration.epoch is not None
                and update_counter.cur_iteration.epoch >= self.stop_at_epoch
            )
        )
