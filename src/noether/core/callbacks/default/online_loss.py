#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from collections import defaultdict

import torch

from noether.core.callbacks.periodic import IntervalType, PeriodicCallback
from noether.core.distributed import all_gather_nograd, all_reduce_mean_nograd
from noether.core.schemas.callbacks import OnlineLossCallbackConfig


class OnlineLossCallback(PeriodicCallback):
    """Callback to track the loss of the model after every gradient accumulation step and log the average loss."""

    def __init__(self, callback_config: OnlineLossCallbackConfig, **kwargs):
        """Initializes the OnlineLossCallback.

        Args:
            callback_config: The configuration for the callback.
            **kwargs: additional arguments passed to the parent class.
        """
        super().__init__(callback_config=callback_config, **kwargs)
        self.verbose = callback_config.verbose
        self.tracked_losses: defaultdict[str, list[torch.Tensor]] = defaultdict(list)

    def _track_after_accumulation_step(self, *, losses, **_) -> None:
        for name, loss in losses.items():
            self.tracked_losses[name].append(loss.detach())

    def _periodic_callback(self, *, interval_type: IntervalType, **_) -> None:
        if interval_type == "eval":
            return  # online losses are only logged during training
        for name, tracked_loss in self.tracked_losses.items():
            mean_loss = all_reduce_mean_nograd(torch.stack(tracked_loss).mean())
            if not self.trainer.config.skip_nan_loss and torch.isnan(mean_loss):
                losses = all_gather_nograd(torch.stack(tracked_loss))
                num_nans = torch.isnan(losses).sum()
                msg = f"encountered nan loss ({num_nans.item()} nans): {losses}"
                self.logger.error(msg)
                raise RuntimeError(msg)
            self.writer.add_scalar(
                key=f"loss/online/{name}/{self.to_short_interval_string()}",
                value=mean_loss,
                logger=self.logger if self.verbose else None,
            )
        self.tracked_losses.clear()
