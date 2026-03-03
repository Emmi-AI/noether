#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import torch
import torch.nn.functional as F

from examples.shared.cfd_automotive.schemas.trainers import AutomotiveAerodynamicsCfdTrainerConfig
from noether.training.trainers import BaseTrainer


class AutomotiveAerodynamicsCFDTrainer(BaseTrainer):
    """Trainer class for to train automative aerodynaimcs CFD for the: AhmedML, DrivaerML and Shapenet-Car Car dataset."""

    def __init__(self, trainer_config: AutomotiveAerodynamicsCfdTrainerConfig, **kwargs):
        """Trainer class for to train automative aerodynaimcs CFD for the: AhmedML, DrivaerML and Shapenet-Car Car dataset.

        Args:
            trainer_config: Configuration for the trainer.
            **kwargs: Additional keyword arguments for the SgdTrainer.

        Raises:
            ValueError: When an output mode is not defined in the loss items.
        """
        super().__init__(
            config=trainer_config,
            **kwargs,
        )

        self.surface_pressure_weight = trainer_config.surface_pressure_weight
        self.surface_friction_weight = trainer_config.surface_friction_weight
        self.volume_velocity_weight = trainer_config.volume_velocity_weight
        self.volume_pressure_weight = trainer_config.volume_pressure_weight
        self.volume_vorticity_weight = trainer_config.volume_vorticity_weight

        self.surface_weight = trainer_config.surface_weight
        self.volume_weight = trainer_config.volume_weight

        loss_items = {
            "surface_pressure": (self.surface_pressure_weight, self.surface_weight),
            "surface_friction": (
                self.surface_friction_weight,
                self.surface_weight,
            ),  # not used for ShapeNet-Car
            "volume_velocity": (self.volume_velocity_weight, self.volume_weight),
            "volume_pressure": (self.volume_pressure_weight, self.volume_weight),  # not used for ShapeNet-Car
            "volume_vorticity": (self.volume_vorticity_weight, self.volume_weight),  # not used for ShapeNet-Car
        }

        self.loss_items = []
        for target_property in self.target_properties:
            if target_property[: -len("_target")] not in loss_items:
                raise ValueError(f"Output mode '{target_property}' is not defined in loss items.")
            self.loss_items.append(
                (
                    target_property[: -len("_target")],
                    loss_items[target_property[: -len("_target")]][0],
                    loss_items[target_property[: -len("_target")]][1],
                )
            )

    def loss_compute(
        self, forward_output: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Given the output of the model and the targets, compute the losses.
        Args:
            forward_output The output of the model, containing the predictions for each output mode.
            targets: Dict containing all target values to compute the loss.

        Returns:
            A dictionary containing the computed losses for each output mode.
        """
        losses: dict[str, torch.Tensor] = {}
        for item, weight, group_weight in self.loss_items:
            if weight > 0 and group_weight > 0 and item in forward_output:
                if f"{item}_target" not in targets:
                    raise ValueError(
                        f"Target for '{item}' not found in targets. Ensure the targets contain the correct keys."
                    )
                losses[f"{item}_loss"] = (
                    F.mse_loss(targets[f"{item}_target"], forward_output[item]) * weight * group_weight
                )
        if len(losses) == 0:
            raise ValueError("No losses computed, check your output keys and loss function.")
        return losses
