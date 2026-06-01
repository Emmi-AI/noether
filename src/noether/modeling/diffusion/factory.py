#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import Annotated, Union

from pydantic import Field

from .base import DiffusionSchedule
from .flow_matching import FlowMatchingConfig, FlowMatchingSchedule

AnyDiffusionScheduleConfig = Annotated[
    Union[FlowMatchingConfig],
    Field(discriminator="kind"),
]
"""Discriminated union of all built-in diffusion schedule configurations.

Pydantic resolves the right variant by inspecting the ``kind`` field. Pair
with :func:`build_schedule` to materialize the schedule object."""


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
