#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import math
from typing import Any

import torch.nn as nn

from noether.core.callbacks.periodic import PeriodicCallback
from noether.core.utils.model import compute_model_norm


class TrainingStabilityCallback(PeriodicCallback):
    """
    A callback that logs the norm of the gradients, the grad scaler scale and the model norm, additionally all the losses after the accumulation step are logged. This can be useful for monitoring
    training and diagnosing issues with exploding or vanishing gradients.
    This callback is not added is added by default to the trainer
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_logged_grad_scaler_scale: float | None = None

    def _zero_grad_ratio(self, model: nn.Module | None, eps=1e-8) -> float:
        """Computes the ratio of gradients that are near zero to the total number of gradients."""

        if model is None or not isinstance(model, nn.Module):
            return 0.0

        total = 0
        near_zero = 0
        for p in model.parameters():
            if p.grad is None:
                continue
            total += 1
            if p.grad.norm() < eps:
                near_zero += 1
        return near_zero / total if total > 0 else 0.0

    # noinspection PyMethodOverriding
    def periodic_callback(self, **_) -> None:
        grad_scaler = self.trainer.grad_scaler
        if not grad_scaler.is_enabled():
            return
        scale = grad_scaler.get_scale()
        if scale != self._last_logged_grad_scaler_scale:
            self.writer.add_scalar("profiling/optim/grad_scaler_scale", scale)
            self._last_logged_grad_scaler_scale = scale

        for cur_name, cur_model in self.model.get_named_models().items():
            optimizer = cur_model.optimizer
            if optimizer is None or optimizer.last_grad_norm is None:
                continue
            norm = optimizer.last_grad_norm.item()
            if math.isfinite(norm):
                self.writer.add_scalar(f"profiling/optim/grad_norm/{cur_name}", norm)
        model_norm = compute_model_norm(self.model).item()
        self.writer.add_scalar("profiling/optim/model_norm", model_norm)
        self.writer.add_scalar("profiling/optim/zero_grad_ratio", self._zero_grad_ratio(self.model))

    def track_after_accumulation_step(self, *, losses, **_) -> None:
        for loss, value in losses.items():
            self.writer.add_scalar(f"profiling/accumulation_step/loss/{loss}", value.item())
