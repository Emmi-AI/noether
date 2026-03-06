#  Copyright © 2025 Emmi AI GmbH. All rights reserved.


from typing import Annotated, ClassVar

from pydantic import Field

from noether.core.schemas.initializers import AnyInitializer
from noether.core.schemas.lib import _RegistryBase
from noether.core.schemas.optimizers import OptimizerConfig


class ModelBaseConfig(_RegistryBase):
    _registry: ClassVar[dict[str, type]] = {}
    _type_field: ClassVar[str] = "kind"

    kind: str | None = None
    """Kind of model to use, i.e. class path"""
    name: str
    """Name of the model. Needs to be unique"""
    optimizer_config: OptimizerConfig | None = None
    """The optimizer configuration to use for training the model. When a model is used for inference only, this can be left as None."""
    initializers: list[Annotated[AnyInitializer, Field(discriminator="kind")]] | None = None
    """List of initializers configs to use for the model."""
    is_frozen: bool | None = False
    """Whether to freeze the model parameters (i.e., not trainable)."""
    forward_properties: list[str] | None = []
    """List of properties to be used as inputs for the forward pass of the model. Only relevant when the train_step of the BaseTrainer is used. When overridden in a class method, this property is ignored."""

    model_config = {"extra": "forbid"}

    @property
    def config_kind(self) -> str:
        """The fully qualified import path for the configuration class."""
        # Use __qualname__ to correctly handle nested classes
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"
