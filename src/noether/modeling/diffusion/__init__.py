#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Diffusion / flow-matching schedules.

Tensor-only flow-matching implementation behind a common
:class:`DiffusionSchedule` ABC. Pair with the configs in
:mod:`noether.core.schemas.diffusion`.

Use :func:`build_schedule` to instantiate the right schedule from a config
that came out of the :data:`~noether.core.schemas.diffusion.AnyDiffusionScheduleConfig`
discriminated union.

Example:

.. testcode::

    import torch
    from noether.core.schemas.diffusion import FlowMatchingConfig
    from noether.modeling.diffusion import build_schedule

    schedule = build_schedule(FlowMatchingConfig()).to("cpu")
    x0 = torch.randn(4, 16)

    def model_fn(xt, t, condition):
        return torch.zeros_like(xt)

    loss = schedule.training_losses(x0, model_fn)
"""

from .base import DiffusionSchedule
from .factory import AnyDiffusionScheduleConfig, build_schedule
from .flow_matching import FlowMatchingConfig, FlowMatchingSchedule

__all__ = [
    "AnyDiffusionScheduleConfig",
    "DiffusionSchedule",
    "FlowMatchingConfig",
    "FlowMatchingSchedule",
    "build_schedule",
]
