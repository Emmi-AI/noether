#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Literal

from noether.core.callbacks.base import CallBackBaseConfig


class BoilerplateCallbackConfig(CallBackBaseConfig):
    kind: str | None = None
    dataset_key: str
    name: Literal["BoilerplateCallback"] = "BoilerplateCallback"
