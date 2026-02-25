#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import torch

from noether.core.schemas.trainers import BaseTrainerConfig
from noether.training.trainers import BaseTrainer


class TrainerConfig(BaseTrainerConfig):
    # custom trainer config options can be added here
    pass


class Trainer(BaseTrainer):
    def __init__(self, trainer_config: TrainerConfig, **kwargs):
        super().__init__(
            config=trainer_config,
            **kwargs,
        )

    def loss_compute(
        self, forward_output: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Implement the loss computation logic here. This method should return")
