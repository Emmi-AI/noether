#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from noether.core.schemas.diffusion import (
    AnyDiffusionScheduleConfig,
    FlowMatchingConfig,
)

from .base import DiffusionSchedule
from .flow_matching import FlowMatchingSchedule


def build_schedule(config: AnyDiffusionScheduleConfig) -> DiffusionSchedule:
    """Instantiate the right :class:`DiffusionSchedule` for ``config``.

    Args:
        config: Any variant of
            :data:`~noether.core.schemas.diffusion.AnyDiffusionScheduleConfig`.

    Returns:
        A :class:`DiffusionSchedule` matching the variant's ``kind``.

    Raises:
        ValueError: If ``config`` is not a recognised schedule config.
    """
    if isinstance(config, FlowMatchingConfig):
        return FlowMatchingSchedule(config)
    raise ValueError(f"Unknown diffusion schedule config: {type(config).__name__}")
