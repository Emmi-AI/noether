#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import torch
import torch.nn.functional as F

from noether.core.schemas.trainers import WeightedMSETrainerConfig
from noether.training.trainers.base import BaseTrainer


class WeightedMSETrainer(BaseTrainer):
    """Generic trainer that computes weighted MSE loss per output field.

    Expects the model forward to return ``dict[str, Tensor]`` with keys matching ``field_weights`` keys, and the batch
    to contain ``<field_name>_target`` keys.
    """

    def __init__(self, trainer_config: WeightedMSETrainerConfig, **kwargs):
        super().__init__(config=trainer_config, **kwargs)

        self.loss_items: list[tuple[str, float]] = []
        for target_prop in self.target_properties:
            field_name = target_prop.removesuffix("_target")
            weight = trainer_config.field_weights.get(field_name)
            if weight is None:
                raise ValueError(
                    f"Target property '{target_prop}' (field '{field_name}') "
                    f"not found in field_weights. Available: {list(trainer_config.field_weights.keys())}"
                )
            self.loss_items.append((field_name, weight))

    def loss_compute(
        self,
        forward_output: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        for field_name, weight in self.loss_items:
            if weight <= 0 or field_name not in forward_output:
                continue
            target_key = f"{field_name}_target"
            if target_key not in targets:
                raise ValueError(f"Target '{target_key}' not found in targets. Available: {list(targets.keys())}")
            losses[f"{field_name}_loss"] = F.mse_loss(targets[target_key], forward_output[field_name]) * weight
        if not losses:
            raise ValueError(
                "No losses computed. Check that field_weights keys match model output keys and target_properties."
            )
        return losses
