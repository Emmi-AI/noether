#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from enum import StrEnum

_MODEL_CLASS_NAMES: dict[str, str] = {
    "transformer": "Transformer",
    "upt": "UPT",
    "ab_upt": "ABUPT",
    "transolver": "Transolver",
}


class ModelChoice(StrEnum):
    TRANSFORMER = "transformer"
    UPT = "upt"
    AB_UPT = "ab_upt"
    TRANSOLVER = "transolver"

    @property
    def class_name(self) -> str:
        return _MODEL_CLASS_NAMES[self.value]

    @property
    def module_name(self) -> str:
        return self.value

    @property
    def schema_module(self) -> str:
        return f"{self.value}_config"

    @property
    def config_class_name(self) -> str:
        return f"{self.class_name}Config"


class DatasetChoice(StrEnum):
    SHAPENET_CAR = "shapenet_car"
    DRIVAERNET = "drivaernet"
    DRIVAERML = "drivaerml"
    AHMEDML = "ahmedml"
    EMMI_WING = "emmi_wing"


class OptimizerChoice(StrEnum):
    ADAMW = "adamw"
    LION = "lion"


class TrackerChoice(StrEnum):
    WANDB = "wandb"
    TRACKIO = "trackio"
    TENSORBOARD = "tensorboard"
    DISABLED = "disabled"


class HardwareChoice(StrEnum):
    GPU = "gpu"
    MPS = "mps"
    CPU = "cpu"
