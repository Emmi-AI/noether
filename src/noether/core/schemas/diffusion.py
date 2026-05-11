#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Configuration schemas for diffusion / flow-matching schedules.

The currently supported paradigm (flow matching) is exposed via the
discriminated union :data:`AnyDiffusionScheduleConfig` keyed on the ``kind``
field. Use it as a typed slot in larger configs, e.g.::

    from noether.core.schemas.diffusion import AnyDiffusionScheduleConfig
    from pydantic import Field


    class MyTrainerConfig(BaseModel):
        diffusion_schedule: AnyDiffusionScheduleConfig = Field(default_factory=FlowMatchingConfig, discriminator="kind")

The matching :func:`noether.modeling.diffusion.build_schedule` factory
instantiates the right :class:`~noether.modeling.diffusion.DiffusionSchedule`
subclass from any variant of the union.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class FlowMatchingConfig(BaseModel):
    """Rectified flow matching with optional minibatch optimal transport.

    Discriminator: ``kind = "flow_matching"``. Linear interpolation path
    ``xt = t * x1 + (1-t) * x0``; the network predicts the velocity
    ``v = x1 - x0``.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["flow_matching"] = "flow_matching"
    continuous_time: bool = True
    """If True, sample t with logit-normal; otherwise uniform on [0, 1]."""
    minibatch_ot: bool = False
    """If True, reorder the noise samples within a minibatch via optimal
    transport against the data (Pooladian et al. 2023). Requires SciPy."""


AnyDiffusionScheduleConfig = Annotated[
    Union[FlowMatchingConfig],
    Field(discriminator="kind"),
]
"""Discriminated union of all built-in diffusion schedule configurations.

Pydantic resolves the right variant by inspecting the ``kind`` field. Pair
with :func:`noether.modeling.diffusion.build_schedule` to materialize the
schedule object."""
