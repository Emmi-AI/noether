#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Union

from .base import InitializerBase, InitializerConfig
from .checkpoint import CheckpointInitializer, CheckpointInitializerConfig
from .previous_run import PreviousRunInitializer, PreviousRunInitializerConfig
from .resume import ResumeInitializer, ResumeInitializerConfig

AnyInitializer = Union[CheckpointInitializerConfig, ResumeInitializerConfig, PreviousRunInitializerConfig]

__all__ = [
    # --- classes:
    "InitializerBase",
    "CheckpointInitializer",
    "PreviousRunInitializer",
    "ResumeInitializer",
    # --- configs:
    "InitializerConfig",
    "CheckpointInitializerConfig",
    "PreviousRunInitializerConfig",
    "ResumeInitializerConfig",
    "AnyInitializer",
]
