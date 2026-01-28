#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from datetime import datetime

from noether.core.callbacks.periodic import PeriodicCallback
from noether.core.schemas.callbacks import CallBackBaseConfig
from noether.core.utils.logging import seconds_to_duration_str
from noether.core.utils.training import UpdateCounter  # fixme?


class ProgressCallback(PeriodicCallback):
    """Callback to print the progress of the training such as number of epochs and updates. Is initialized by the BaseTrainer and should not be added manually to the trainer's callbacks."""

    def __init__(self, callback_config: CallBackBaseConfig, **kwargs):
        super().__init__(callback_config=callback_config, **kwargs)
        self._start_time: datetime | None = None
        self._last_log_time: datetime | None = None
        self._last_log_samples = 0

    def before_training(self, **_) -> None:
        self._start_time = self._last_log_time = datetime.now()

    # noinspection PyMethodOverriding
    def periodic_callback(self, *, interval_type, update_counter: UpdateCounter, **_) -> None:
        if self.trainer.end_checkpoint.epoch is not None:
            total_updates = self.trainer.end_checkpoint.epoch * update_counter.updates_per_epoch
        elif self.trainer.end_checkpoint.update is not None:
            total_updates = self.trainer.end_checkpoint.update
        elif self.trainer.end_checkpoint.sample is not None and update_counter.cur_iteration.sample is not None:
            total_updates = update_counter.cur_iteration.sample // update_counter.effective_batch_size
        else:
            raise NotImplementedError

        if interval_type == "epoch":
            self.logger.info(
                f"Epoch {update_counter.cur_iteration.epoch}/{self.trainer.end_checkpoint.epoch} "
                f"({update_counter.cur_iteration})"
            )
        elif interval_type == "update":
            self.logger.info(
                f"Update {update_counter.cur_iteration.update}/{total_updates} ({update_counter.cur_iteration})"
            )
        elif interval_type == "sample":
            self.logger.info(
                f"Sample {update_counter.cur_iteration.sample}/{self.trainer.end_checkpoint.sample} "
                f"({update_counter.cur_iteration})"
            )

        assert self._last_log_time is not None
        assert self._start_time is not None
        assert update_counter.cur_iteration.sample is not None
        assert update_counter.cur_iteration.update is not None

        now = datetime.now()
        seconds_since_last_log = (now - self._last_log_time).total_seconds()
        samples_since_last_log = update_counter.cur_iteration.sample - self._last_log_samples
        updates_since_last_log = samples_since_last_log // update_counter.effective_batch_size
        if self._last_log_samples == 0:
            progress = update_counter.cur_iteration.update / total_updates
        else:
            # subtract first interval to give better estimate
            total_updates -= updates_since_last_log
            cur_update = update_counter.cur_iteration.update - updates_since_last_log
            progress = cur_update / total_updates
        estimated_duration = (now - self._start_time) / progress
        self.logger.info(
            f"ETA: {(self._start_time + estimated_duration).strftime('%m.%d %H.%M.%S')} "
            f"estimated_duration: {seconds_to_duration_str(estimated_duration.total_seconds())} "
            f"time_since_last_log: {seconds_to_duration_str(seconds_since_last_log)} "
            f"time_per_update: {seconds_to_duration_str(seconds_since_last_log / updates_since_last_log)} "
        )
        # reset after first log because first few updates take longer which skew the ETA
        if self._last_log_samples == 0:
            self._start_time = now
        self._last_log_time = now
        if update_counter.cur_iteration.sample is not None:
            self._last_log_samples = update_counter.cur_iteration.sample
