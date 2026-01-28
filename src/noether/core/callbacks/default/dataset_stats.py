#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.callbacks.base import CallbackBase


class DatasetStatsCallback(CallbackBase):
    """A callback that logs the length of each dataset in the data container. Is initialized by the BaseTrainer and should not be added manually to the trainer's callbacks."""

    def before_training(self, **_) -> None:
        for dataset_key, dataset in self.data_container.datasets.items():
            self.tracker.set_summary(key=f"ds_stats/{dataset_key}/len", value=len(dataset))

        dataset_str = ", ".join(
            f"{dataset_key}={len(dataset)}" for dataset_key, dataset in self.data_container.datasets.items()
        )
        self.logger.info(f"Dataset lengths: {dataset_str}")
