#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections import defaultdict

import torch

from noether.core.callbacks.periodic import PeriodicCallback
from noether.core.distributed import all_gather_nograd
from noether.core.distributed.gather import reduce_max_nograd, reduce_mean_nograd
from noether.core.schemas.callbacks import CallBackBaseConfig
from noether.core.utils.logging import tensor_like_to_string


class TrainTimeCallback(PeriodicCallback):
    """Callback to log the time spent on dataloading. Is initialized by the :class:`~noether.training.trainers.BaseTrainer` and should not be added manually to the trainer's callbacks."""

    def __init__(self, callback_config: CallBackBaseConfig, **kwargs):
        super().__init__(callback_config=callback_config, **kwargs)
        self.train_times: dict[str, list[float]] = defaultdict(list)
        self.total_train_times: dict[str, float] = {}

    def track_after_update_step(self, *, times: dict[str, float], **_) -> None:
        for k, v in times.items():
            self.train_times[k].append(v)

    def periodic_callback(self, **_) -> None:
        for k, v in self.train_times.items():
            arr = torch.tensor(v)
            mean = torch.mean(arr) if len(arr) > 0 else torch.tensor(0.0)
            max_v = torch.max(arr) if len(arr) > 0 else torch.tensor(0.0)

            # accumulate total for after_training
            if k not in self.total_train_times:
                self.total_train_times[k] = 0.0
            self.total_train_times[k] += float(torch.sum(arr))

            # we only reduce to rank=0 because we only log and track at rank=0.
            mean_gathered = reduce_mean_nograd(mean)
            max_gathered = reduce_max_nograd(max_v)
            self.writer.add_scalar(
                key=f"train_time/{k}/mean/{self.to_short_interval_string()}",
                value=mean_gathered,
                logger=self.logger,
            )
            self.writer.add_scalar(
                key=f"train_time/{k}/max/{self.to_short_interval_string()}",
                value=max_gathered,
                logger=self.logger,
            )

        self.train_times.clear()

    def after_training(self, **_) -> None:
        for k, v in self.total_train_times.items():
            total_time = all_gather_nograd(v)
            self.logger.info(f"{k}: total={tensor_like_to_string(total_time)} [sec] in total")
