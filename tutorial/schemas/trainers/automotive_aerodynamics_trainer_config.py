#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Union

from pydantic import Field

from noether.core.schemas.callbacks import CallbacksConfig
from noether.core.schemas.trainers import BaseTrainerConfig
from tutorial.schemas.callbacks import TutorialCallbacksConfig

AllCallbacks = Union[
    TutorialCallbacksConfig, CallbacksConfig
]  # custom callbacks need to be added here to one union type with the base noether CallbacksConfig


class AutomotiveAerodynamicsCfdTrainerConfig(BaseTrainerConfig):
    surface_weight: float = 1.0
    """ Weight of the predicted values on the surface mesh. Defaults to 1.0.."""
    volume_weight: float = 1.0
    """Weight of the predicted values in the volume. Defaults to 1.0."""
    surface_pressure_weight: float = 1.0
    """Weight of the predicted values for the surface pressure. Defaults to 1.0."""
    surface_friction_weight: float = 0.0
    """Weight of the predicted values for the surface wall shear stress. Defaults to 0.0."""
    volume_velocity_weight: float = 1.0
    """Weight of the predicted values for the volume velocity. Defaults to 1.0."""
    volume_pressure_weight: float = 0.0
    """Weight of the predicted values for the volume total pressure coefficient. Defaults to 0.0."""
    volume_vorticity_weight: float = 0.0
    """Weight of the predicted values for the volume vorticity. Defaults to 0.0."""
    use_physics_features: bool = False
    """ If true, additional features are used next to the input coordidates (i.e., SDF, surfacer normals, etc.). Defaults to False."""
    callbacks: list[AllCallbacks] | None = Field(
        ...,
    )  # we need to override this to include the tutorial callback; ideally we would not need to do this? But I also don't know how to fix this otherwise?
