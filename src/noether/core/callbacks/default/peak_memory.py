#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch.cuda

from noether.core.callbacks.periodic import PeriodicCallback


class PeakMemoryCallback(PeriodicCallback):
    """Callback to log the peak memory usage of the model. Is initialized by the :class:`~noether.training.trainers.BaseTrainer` and should not be added manually to the trainer's callbacks."""

    def periodic_callback(self, **__) -> None:
        if str(self.model.device) != "cuda":
            return
        self.logger.info(f"max_memory_allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
