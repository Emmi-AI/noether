#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import torch
from pydantic import Field
from torch import Tensor

from noether.core.schemas import BaseTrainerConfig
from noether.training.trainers import BaseTrainer
from noether.training.trainers.types import LossResult


class UQTrainerConfig(BaseTrainerConfig):
    """Trainer config for Gaussian NLL heteroscedastic training."""

    field_weights: dict[str, float] = Field(
        ..., description="Per-field loss weights, e.g. {'surface_pressure': 1.0, 'volume_velocity': 1.0}"
    )
    nll_loss_weight: float = Field(1.0, ge=0.0, description="Weight for Gaussian NLL loss component")
    mse_loss_weight: float = Field(0.1, ge=0.0, description="Weight for MSE loss on mean predictions (stability)")
    variance_regularization: float = Field(0.01, ge=0.0, description="Regularization weight on log-variance")
    warmup_epochs_mse_only: int = Field(0, ge=0, description="Train with MSE only for this many epochs before NLL")
    use_physics_features: bool = Field(False, description="Whether to use physics features as model input")


class UQTrainer(BaseTrainer):
    """Trainer for heteroscedastic AB-UPT with Gaussian NLL loss.

    Expects model forward output to contain '{field}_mean' and '{field}_log_var' keys.
    Targets in the batch should follow the '{field}_target' convention.
    """

    def __init__(self, trainer_config: UQTrainerConfig, **kwargs):
        super().__init__(config=trainer_config, **kwargs)

    def loss_compute(self, forward_output: dict[str, Tensor], targets: dict[str, Tensor]) -> LossResult:
        config: UQTrainerConfig = self.config  # type: ignore[assignment]
        current_epoch = self.update_counter.cur_iteration.epoch if self.update_counter.cur_iteration else 0
        use_nll = current_epoch >= config.warmup_epochs_mse_only

        losses: dict[str, Tensor] = {}

        for field_name, weight in config.field_weights.items():
            if weight <= 0:
                continue

            # Match field_weights key (e.g. "surface_pressure") to model output key
            # Model outputs: "surface_pressure_mean", "surface_pressure_log_var"
            mean_key = f"{field_name}_mean"
            if mean_key not in forward_output:
                continue

            log_var_key = f"{field_name}_log_var"
            target_key = f"{field_name}_target"

            if target_key not in targets:
                continue

            mean = forward_output[mean_key]
            target = targets[target_key]

            # MSE on mean predictions (named _loss to match baseline for comparison)
            mse = torch.nn.functional.mse_loss(mean, target)
            losses[f"{field_name}_loss"] = mse * weight * config.mse_loss_weight

            # Gaussian NLL (only after warmup)
            if use_nll and log_var_key in forward_output:
                log_var = forward_output[log_var_key]
                # NLL = 0.5 * (log_var + (target - mean)^2 / exp(log_var))
                nll = 0.5 * (log_var + (target - mean).pow(2) * torch.exp(-log_var))
                losses[f"{field_name}_nll"] = nll.mean() * weight * config.nll_loss_weight

                # Variance regularization: penalize extreme log-variances
                if config.variance_regularization > 0:
                    var_reg = log_var.pow(2).mean()
                    losses[f"{field_name}_var_reg"] = var_reg * config.variance_regularization

        return losses
