#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from typing import Literal

import torch
import torch.nn.functional as F

from noether.core.callbacks.periodic import PeriodicDataIteratorCallback
from noether.core.schemas.callbacks import PeriodicDataIteratorCallbackConfig


class DevelopmentCallbackConfig(PeriodicDataIteratorCallbackConfig):
    """Configuration for the DevelopmentCallback."""

    name: Literal["DevelopmentCallback"] = "DevelopmentCallback"
    """Name of the callback, used for logging and identification purposes."""
    forward_properties: list[str] = []


class DevelopmentCallback(PeriodicDataIteratorCallback):
    """A simple callback for development purposes that prints the batch data during training."""

    def __init__(self, callback_config: DevelopmentCallbackConfig, **kwargs):
        super().__init__(callback_config=callback_config, **kwargs)
        self.config = callback_config

    def process_data(self, batch: dict[str, torch.Tensor], **kwargs) -> None:
        loss = F.mse_loss(
            self.model(**{prop: batch[prop] for prop in self.config.forward_properties})["output"], batch["y"]
        )

        return {"mse_loss": loss.item()}
