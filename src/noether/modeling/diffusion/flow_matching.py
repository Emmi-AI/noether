#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
from torch import Tensor

from noether.core.schemas.diffusion import FlowMatchingConfig

from .base import DiffusionSchedule


class FlowMatchingSchedule(DiffusionSchedule):
    """Rectified flow matching with optional minibatch optimal transport.

    Linear interpolation path ``xt = t * x1 + (1-t) * x0``; the network
    predicts the velocity ``v = x1 - x0``. Logit-normal time sampling for
    training when ``continuous_time=True``.
    """

    def __init__(self, config: FlowMatchingConfig):
        self.config = config

    def _sample_time(self, bs: int, device: torch.device) -> Tensor:
        if self.config.continuous_time:
            # Logit-normal time sampling
            return torch.sigmoid(torch.randn((bs,), device=device))
        return torch.rand((bs,), device=device)

    def _apply_ot(self, x0: Tensor, x1: Tensor) -> Tensor:
        """Optimal transport reordering of ``x0`` against ``x1`` within the current batch."""
        import scipy.optimize

        bs = x0.shape[0]
        if bs == 1:
            warnings.warn("Optimal transport reordering is a no-op for batch size 1.")
            return x0
        with torch.no_grad():
            x0_flat = x0.reshape(bs, -1)
            x1_flat = x1.reshape(bs, -1)
            cost = torch.cdist(x0_flat, x1_flat).cpu().numpy()
            row_ind, _ = scipy.optimize.linear_sum_assignment(cost)
            return x0[torch.tensor(row_ind, device=x0.device)]

    def noise_pair(self, x0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Noise clean data ``x0`` at time ``t``.

        Returns:
            Tuple ``(xt, target_velocity)``.
        """
        t_expand = t.view(-1, *([1] * (x0.ndim - 1)))
        x0_noise = torch.randn_like(x0)
        if self.config.minibatch_ot:
            x0_noise = self._apply_ot(x0_noise, x0)
        xt = t_expand * x0 + (1.0 - t_expand) * x0_noise
        target_v = x0 - x0_noise
        return xt, target_v

    def training_losses(self, x0, model_fn, condition=None):
        bs = x0.shape[0]
        t = self._sample_time(bs, x0.device)
        xt, target_v = self.noise_pair(x0, t)
        pred = model_fn(xt, t, condition)
        return F.mse_loss(pred, target_v)

    @torch.no_grad()
    def sample(self, shape, model_fn, condition=None, steps=10):
        x = torch.randn(shape, device=self.device)
        t_steps = torch.linspace(0.0, 1.0, steps + 1, device=self.device)

        for i in range(steps):
            dt = t_steps[i + 1] - t_steps[i]
            t_batch = torch.full((shape[0],), t_steps[i].item(), device=self.device)
            v = model_fn(x, t_batch, condition)
            x = x + v * dt

        return x

    def training_losses_joint(self, x0_list, model_fn, condition=None):
        """Joint flow-matching loss over multiple clean tensors sharing batch dim.

        All tensors are noised to the SAME ``t`` per example. ``model_fn``
        receives ``(xt_list, t, condition)`` and must return a list of
        velocity predictions aligned with ``x0_list``. Returns mean MSE
        across tensors.
        """
        assert len(x0_list) > 0
        bs = x0_list[0].shape[0]
        device = x0_list[0].device
        for x in x0_list:
            assert x.shape[0] == bs, "all x0 must share batch dim"

        t = self._sample_time(bs, device)

        xt_list = []
        target_list = []
        for x1 in x0_list:
            xt, target_v = self.noise_pair(x1, t)
            xt_list.append(xt)
            target_list.append(target_v)

        preds = model_fn(xt_list, t, condition)
        assert len(preds) == len(x0_list)
        losses = [F.mse_loss(p, tgt) for p, tgt in zip(preds, target_list, strict=True)]
        return sum(losses) / len(losses)

    @torch.no_grad()
    def sample_joint(self, shapes, model_fn, condition=None, steps=10):
        """Joint Euler sampling over multiple tensors sharing batch dim.

        ``model_fn`` receives ``(xt_list, t, condition)`` and must return a
        list of velocities aligned with ``shapes``. Returns the list of
        clean samples.
        """
        assert len(shapes) > 0
        bs = shapes[0][0]
        for s in shapes:
            assert s[0] == bs, "all shapes must share batch dim"

        xs = [torch.randn(shape, device=self.device) for shape in shapes]
        t_steps = torch.linspace(0.0, 1.0, steps + 1, device=self.device)

        for i in range(steps):
            dt = t_steps[i + 1] - t_steps[i]
            t_batch = torch.full((bs,), t_steps[i].item(), device=self.device)
            vs = model_fn(xs, t_batch, condition)
            xs = [x + v * dt for x, v in zip(xs, vs, strict=True)]

        return xs
