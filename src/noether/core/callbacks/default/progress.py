#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from datetime import datetime
from typing import override

from noether.core.callbacks.periodic import PeriodicCallback
from noether.core.utils.logging import seconds_to_duration_str
from noether.core.utils.training import UpdateCounter


class ProgressCallback(PeriodicCallback):
    """Callback to print the progress of the training such as number of epochs and updates.

    This callback is initialized by the :class:`~noether.training.trainers.BaseTrainer` and should not be added
    manually to the trainer's callbacks.
    """

    _start_time: datetime | None = None
    _last_log_time: datetime | None = None
    _last_track_time: datetime | None = None
    _last_log_samples = 0
    _MIN_LOG_INTERVAL_S = 300

    def before_training(self, **_) -> None:
        self._start_time = self._last_log_time = self._last_track_time = datetime.now()

    # noinspection PyMethodOverriding
    def periodic_callback(self, *, interval_type, update_counter: UpdateCounter, **_) -> None:
        total_updates = self.trainer.total_training_updates
        done_str = ""
        if total_updates is not None and update_counter.cur_iteration.update is not None:
            done_str = f"{(update_counter.cur_iteration.update / total_updates):.0%} done."
        if interval_type == "epoch":
            self.logger.info(
                f"{done_str} Epoch {update_counter.cur_iteration.epoch}/{self.trainer.end_checkpoint.epoch} "
                f"({update_counter.cur_iteration})"
            )
        elif interval_type == "update":
            self.logger.info(
                f"{done_str} Update {update_counter.cur_iteration.update}/{total_updates} ({update_counter.cur_iteration})"
            )
        elif interval_type == "sample":
            self.logger.info(
                f"{done_str} Sample {update_counter.cur_iteration.sample}/{self.trainer.end_checkpoint.sample} "
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
        eta_utc = (self._start_time + estimated_duration).astimezone().replace(microsecond=0).isoformat()
        self.logger.info(
            f"Estimated end time (UTC): {eta_utc} "
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

    @override
    def track_after_update_step(self, *, update_counter: UpdateCounter, **_) -> None:
        if self._last_track_time is None or self._last_log_time is None:
            return
        time_since_last_log = (datetime.now() - self._last_log_time).total_seconds()
        time_since_last_track = (datetime.now() - self._last_track_time).total_seconds()
        # Track at least every few seconds to ensure that we track something even if logging intervals are long (e.g., log_every_n_epochs=1)
        if time_since_last_log > self._MIN_LOG_INTERVAL_S and time_since_last_track > self._MIN_LOG_INTERVAL_S:
            self.periodic_callback(
                interval_type="update",
                update_counter=update_counter,
            )
            self._last_track_time = datetime.now()
