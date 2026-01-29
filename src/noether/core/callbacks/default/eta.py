#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
import sys
from datetime import UTC, datetime, timedelta

import numpy as np

from noether.core.callbacks.periodic import PeriodicCallback
from noether.core.distributed import is_rank0
from noether.core.schemas.callbacks import CallBackBaseConfig
from noether.core.utils.logging import short_number_str
from noether.core.utils.training import UpdateCounter


class EtaCallback(PeriodicCallback):
    """Callback to print the progress and estimated duration until the periodic callback will be invoked.

    Also counts up the current epoch/update/samples and provides the average update duration. Only used in "unmanaged"
    runs, i.e., it is not used when the run was started via SLURM.

    This callback is initialized by the :class:`~noether.training.trainers.BaseTrainer` and should not be added
    manually to the trainer's callbacks.
    """

    class LoggerWasCalledHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.was_called = False

        def emit(self, _):
            self.was_called = True

    def __init__(self, callback_config: CallBackBaseConfig, **kwargs):
        super().__init__(callback_config=callback_config, **kwargs)
        self.total_time = 0.0
        self.time_since_last_log = 0.0
        self.handler = self.LoggerWasCalledHandler()
        self._start_time: datetime | None = None

    def before_training(self, *, update_counter: UpdateCounter) -> None:
        assert is_rank0(), "only use EtaCallback on rank0 process"
        self.epoch_format = f"{int(np.log10(max(1, update_counter.end_iteration.epoch or 0))) + 1}d"
        self.update_format = f"{int(np.log10(update_counter.end_iteration.update or 1)) + 1}d"
        self.every_n_epochs_format = f"{int(np.log10(self.every_n_epochs)) + 1}d" if self.every_n_epochs else None
        self.every_n_updates_format = f"{int(np.log10(self.every_n_updates)) + 1}d" if self.every_n_updates else None

        self.updates_per_log_interval_format: str | None = None
        if self.every_n_epochs:
            self.updates_per_log_interval_format = f"{int(np.log10(update_counter.updates_per_epoch)) + 1}d"
        elif self.every_n_updates:
            self.updates_per_log_interval_format = self.every_n_updates_format
        elif self.every_n_samples:
            self.updates_per_every_n_samples = np.ceil(self.every_n_samples / update_counter.effective_batch_size)
            self.updates_per_log_interval_format = f"{int(np.log10(self.updates_per_every_n_samples)) + 1}d"
        self._start_time = datetime.now()

    def track_after_update_step(self, *, update_counter: UpdateCounter, times) -> None:
        cur_epoch = update_counter.cur_iteration.epoch
        assert cur_epoch is not None
        cur_update = update_counter.cur_iteration.update
        assert cur_update is not None
        cur_sample = update_counter.cur_iteration.sample
        assert cur_sample is not None

        now = datetime.now()
        # reset time_since_last_log on new log interval
        if self._should_log_after_epoch(update_counter.cur_iteration) and update_counter.is_full_epoch:
            self.time_since_last_log = 0.0
        if self._should_log_after_update(update_counter.cur_iteration):
            self.time_since_last_log = 0.0
        if self._should_log_after_sample(update_counter.cur_iteration, update_counter.effective_batch_size):
            self.time_since_last_log = 0.0

        if self.every_n_epochs:
            last_epoch = self.every_n_epochs * (cur_epoch // self.every_n_epochs)
            updates_at_last_log = last_epoch * update_counter.updates_per_epoch
            updates_since_last_log = cur_update - updates_at_last_log
            updates_per_log_interval = self.every_n_epochs * update_counter.updates_per_epoch
            if updates_since_last_log == 0:
                updates_since_last_log = updates_per_log_interval
        elif self.every_n_updates:
            updates_since_last_log = cur_update % self.every_n_updates
            updates_per_log_interval = self.every_n_updates
        elif self.every_n_samples:
            samples_since_last_log = cur_sample % self.every_n_samples
            samples_at_last_log = cur_sample - samples_since_last_log
            updates_at_last_log = samples_at_last_log // update_counter.effective_batch_size
            superflous_samples_at_last_log = samples_at_last_log % update_counter.effective_batch_size
            updates_since_last_log = cur_update - updates_at_last_log
            samples_for_cur_log_interval = self.every_n_samples - superflous_samples_at_last_log
            updates_per_log_interval = int(np.ceil(samples_for_cur_log_interval / update_counter.effective_batch_size))
        else:
            updates_since_last_log = None
            updates_per_log_interval = None

        # add time
        time_increment = times["data_time"] + times["update_time"]
        self.total_time += time_increment
        self.time_since_last_log += time_increment
        average_update_time = self.total_time / cur_update

        logstr = (
            f"E {format(cur_epoch, self.epoch_format)}/{update_counter.end_iteration.epoch} | "
            f"U {format(cur_update, self.update_format)}/{update_counter.end_iteration.update} | "
            f"S {short_number_str(cur_sample):>6}/"
            f"{short_number_str(update_counter.end_iteration.sample) if update_counter.end_iteration.sample is not None else ''} | "
        )
        # log interval ETA
        if (
            self.updates_per_log_interval_format is not None
            and updates_since_last_log is not None
            and updates_per_log_interval is not None
        ):
            updates_till_next_log = updates_per_log_interval - updates_since_last_log
            time_delta_next_log = timedelta(seconds=updates_till_next_log * average_update_time)
            next_log_eta = now + time_delta_next_log
            # convert to datetime for formatting
            past_next_log_time = datetime.fromtimestamp(self.time_since_last_log, tz=UTC)
            time_till_next_log = datetime.fromtimestamp(time_delta_next_log.total_seconds(), tz=UTC)
            logstr += (
                f"next_log {format(updates_since_last_log, self.updates_per_log_interval_format)}/"
                f"{format(updates_per_log_interval, self.updates_per_log_interval_format)} | "
                f"next_log_eta {next_log_eta.strftime('%H:%M:%S')} "
                f"({time_till_next_log.strftime('%M:%S')}->{past_next_log_time.strftime('%M:%S')}) | "
            )
        logstr += (
            # f"training_eta {training_eta.strftime('%d-%H:%M:%S')} "
            # f"({seconds_to_duration_str(remaining_training_time.total_seconds())}->"
            # f"{seconds_to_duration_str(past_training_time.total_seconds())}) | "
            f"avg_update {average_update_time:.2f}s"
        )
        if self.handler.was_called:
            print(logstr, file=sys.stderr)
            self.handler.was_called = False
        else:
            print(logstr, end="\r", file=sys.stderr)

    def periodic_callback(self, *, interval_type, **_) -> None:
        if interval_type == "update":
            print(file=sys.stderr)

    def after_training(self, **_) -> None:
        logging.getLogger().removeHandler(self.handler)
