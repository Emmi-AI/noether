#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import torch.nn.functional as F

from noether.training.trainers import BaseTrainer
from noether.training.trainers.base import BaseTrainerConfig


class DevelopmentTrainerConfig(BaseTrainerConfig):
    pass


class DevelopmentTrainer(BaseTrainer):
    def __init__(self, trainer_config: DevelopmentTrainerConfig, **kwargs):
        super().__init__(
            config=trainer_config,
            **kwargs,
        )

        self.config = trainer_config

    def loss_compute(self, forward_output: dict[str, any], targets: dict[str, any]) -> dict[str, any]:
        loss = F.mse_loss(forward_output["output"], targets["y"])
        return {"loss": loss}
