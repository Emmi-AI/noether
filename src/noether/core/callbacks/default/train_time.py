#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import numpy as np

from noether.core.callbacks.periodic import PeriodicCallback
from noether.core.distributed import all_gather_nograd
from noether.core.schemas.callbacks import CallBackBaseConfig
from noether.core.utils.logging import tensor_like_to_string


class TrainTimeCallback(PeriodicCallback):
    """Callback to log the time spent on dataloading. Is initialized by the :class:`~noether.training.trainers.BaseTrainer` and should not be added manually to the trainer's callbacks."""
    def __init__(self, callback_config: CallBackBaseConfig, **kwargs):
        super().__init__(callback_config=callback_config, **kwargs)
        self.train_data_times: list[float] = []
        self.total_train_data_time = 0.0

    def track_after_update_step(self, *, times, **_) -> None:
        self.train_data_times.append(times["data_time"])

    def periodic_callback(self, **_) -> None:
        sum_data_time = np.sum(self.train_data_times)
        mean_data_time = sum_data_time / len(self.train_data_times)
        self.total_train_data_time += sum_data_time
        self.train_data_times.clear()

        # gather for all devices
        mean_data_times = all_gather_nograd(mean_data_time)
        if len(mean_data_times) == 1:
            self.logger.info(f"Waited {tensor_like_to_string(mean_data_times)[1:-1]} [sec] for dataloading")
        else:
            self.logger.info(f"Waited {tensor_like_to_string(mean_data_times)} [sec] for dataloading")

    def after_training(self, **_) -> None:
        total_data_time = all_gather_nograd(self.total_train_data_time)
        self.logger.info(f"Waited {tensor_like_to_string(total_data_time)} [sec] for dataloading in total")
