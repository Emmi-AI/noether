#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from enum import StrEnum


class TrackerChoice(StrEnum):
    WANDB = "wandb"
    TRACKIO = "trackio"
    TENSORBOARD = "tensorboard"
    DISABLED = "disabled"


class HardwareChoice(StrEnum):
    GPU = "gpu"
    MPS = "mps"
    CPU = "cpu"
