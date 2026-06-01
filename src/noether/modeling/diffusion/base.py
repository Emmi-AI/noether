#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Self

import torch
from torch import Tensor


class DiffusionSchedule(abc.ABC):
    """Abstract base for diffusion paradigms.

    All schedule state (alphas, sigmas, etc.) is stored as plain tensors.
    Call :meth:`to` before use to move the buffers to the target device.
    """

    device: torch.device = torch.device("cpu")

    def to(self, device: torch.device | str) -> Self:
        """Move every :class:`~torch.Tensor` attribute to ``device`` in place.

        Returns ``self`` for chainability.
        """
        self.device = torch.device(device)
        for attr in vars(self):
            val = getattr(self, attr)
            if isinstance(val, Tensor):
                setattr(self, attr, val.to(self.device))
        return self

    @abc.abstractmethod
    def training_losses(
        self,
        x0: Tensor,
        model_fn: Callable[[Tensor, Tensor, Tensor | None], Tensor],
        condition: Tensor | None = None,
    ) -> Tensor:
        """Compute scalar training loss given clean samples ``x0``.

        Args:
            x0: Clean training samples.
            model_fn: Callable with signature
                ``(noisy_input, timestep_or_sigma, condition) -> prediction``.
            condition: Optional conditioning tensor passed through to ``model_fn``.

        Returns:
            Scalar training loss.
        """

    @abc.abstractmethod
    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],
        model_fn: Callable[[Tensor, Tensor, Tensor | None], Tensor],
        condition: Tensor | None = None,
        steps: int = 50,
    ) -> Tensor:
        """Generate samples from noise.

        Args:
            shape: Output tensor shape.
            model_fn: Callable with signature
                ``(noisy_input, timestep_or_sigma, condition) -> prediction``.
            condition: Optional conditioning tensor.
            steps: Number of solver steps.

        Returns:
            Clean samples ``x0`` of shape ``shape``.
        """
