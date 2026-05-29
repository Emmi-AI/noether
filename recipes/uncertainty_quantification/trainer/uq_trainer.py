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
    variance_regularization: float = Field(0.01, ge=0.0, description="Regularization weight on log-variance")
    warmup_epochs_mse_only: int = Field(0, ge=0, description="Train with MSE only for this many epochs before NLL")
    use_physics_features: bool = Field(False, description="Whether to use physics features as model input")
    beta_nll: float = Field(
        0.0, ge=0.0, le=1.0, description="β-NLL weighting (Seitzer 2022). 0=NLL, 1=MSE-like gradient"
    )
    mse_loss_weight: float = Field(
        0.1, ge=0.0, description="Weight for MSE loss component, used during warmup and if β_NLL > 0"
    )


class UQTrainer(BaseTrainer):
    """Trainer for heteroscedastic AB-UPT with Gaussian NLL loss.

    Expects model forward output to contain '{field}_mean' and '{field}_log_var' keys.
    Targets in the batch should follow the '{field}_target' convention.
    """

    def __init__(self, trainer_config: UQTrainerConfig, **kwargs):
        self.config: UQTrainerConfig  # type hint for self.config
        super().__init__(config=trainer_config, **kwargs)

    def loss_compute(self, forward_output: dict[str, Tensor], targets: dict[str, Tensor]) -> LossResult:
        current_epoch = self.update_counter.cur_iteration.epoch if self.update_counter.cur_iteration else 0
        use_nll = current_epoch >= self.config.warmup_epochs_mse_only
        losses: dict[str, Tensor] = {}

        for field_name, weight in self.config.field_weights.items():
            if weight > 0 and f"{field_name}_mean" in forward_output and f"{field_name}_target" in targets:
                mean_key = f"{field_name}_mean"
                log_var_key = f"{field_name}_log_var"
                target_key = f"{field_name}_target"

                mean = forward_output[mean_key]
                target = targets[target_key]

                sq_err = (mean - target).pow(2)
                if self.config.mse_loss_weight > 0.0:
                    losses[f"{field_name}_regression"] = sq_err.mean() * self.config.mse_loss_weight

                if use_nll and log_var_key in forward_output:
                    # Warmup phase, or no log-variance head: train mean with MSE only.
                    log_var = forward_output[log_var_key]
                    nll = 0.5 * (log_var + sq_err * torch.exp(-log_var))

                    if self.config.beta_nll > 0:
                        # σ^(2β) = exp(β · log_var), detached so it only reweights the loss
                        beta_weight = torch.exp(self.config.beta_nll * log_var).detach()
                        nll = nll * beta_weight

                    losses[f"{field_name}_nll"] = nll.mean() * weight

                    # One-sided: penalize overconfidence (log_var < 0) only.
                    # A symmetric log_var.pow(2) would also pull σ² → 1 regardless of data scale.
                    if self.config.variance_regularization > 0:
                        var_reg = log_var.clamp(max=0).pow(2).mean()
                        losses[f"{field_name}_var_reg"] = var_reg * weight * self.config.variance_regularization

        return losses
