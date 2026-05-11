#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from pydantic import BaseModel

from noether.core.schemas import BaseTrainerConfig  # noqa: F401


class CheckpointConfig(BaseModel):
    epoch: int | None = None
    update: int | None = None
    sample: int | None = None
