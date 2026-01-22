#  Copyright © 2025 Emmi AI GmbH. All rights reserved.


import torch
import torch.nn.functional as F

from noether.training.trainers import BaseTrainer, TrainerResult


class TestTrainer(BaseTrainer):
    def train_step(self, batch: dict[str, torch.Tensor], model: torch.nn.Module) -> TrainerResult:
        """Forward method of the trainer that runs a forward pass on the model and computes the loss.

        Args:
            batch: dict with tensors for the forward pass and the loss computation.
            model: Model instance to run the forward pass on.

        Returns:
            TrainerResult containing the total loss.
        """
        # prepare data
        x = batch["x"]
        target = batch["y"]

        # forward
        y_hat = model(x)

        # calculate loss
        loss = F.cross_entropy(y_hat, target)

        return TrainerResult(total_loss=loss)
