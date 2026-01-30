#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.callbacks.early_stoppers.base import EarlyStopperBase
from noether.core.schemas.callbacks import CallBackBaseConfig, FixedEarlyStopperConfig
from noether.core.utils.training import UpdateCounter


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
