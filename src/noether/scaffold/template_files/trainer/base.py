#  Copyright © 2025 Emmi AI GmbH. All rights reserved.


import torch
import torch.nn.functional as F

from noether.training.trainers import BaseTrainer
from noether.training.trainers.types import TrainerResult


class BoilerplateTrainer(BaseTrainer):
    """A base trainer implementation for the `Noether` framework that runs a simple forward pass and computes a loss value.
    This implementation overrides the `train_step` method to defined in the `BaseTrainer` class.
    However, one could also use the default implementation of the `BaseTrainer` class which performs a similar a forward pass and and the user needs to implement the `compute_loss` method instead.
    """

    @staticmethod
    def train_step(batch: dict[str, torch.Tensor], model: torch.nn.Module) -> TrainerResult:
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
