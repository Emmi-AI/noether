#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Self

import torch
from torch.amp.grad_scaler import GradScaler

from noether.core.models.base import ModelBase
from noether.core.models.model import Model
from noether.core.providers.path import PathProvider
from noether.core.schemas.models import ModelBaseConfig
from noether.core.utils.training import UpdateCounter

if TYPE_CHECKING:
    from noether.data.container import DataContainer


class CompositeModel(ModelBase):
    """A composite model that consists of multiple submodels of type Model. By having multiple submodels, each model can have its own optimizer and learning rate scheduler, from weights etc.
    This is useful for multi-component models,

    A composite model must implement the `submodels` property, which returns a dictionary of submodel names to submodel instances.

    Example code (dummy code):

    .. code-block:: python

        from noether.core.models.composite import CompositeModel
        from somewhere import MyModel1, MyModel2

        class MyCompositeModel(CompositeModel):
            def __init__(self, model_config: MyCompositeModelConfig, update_counter: UpdateCounter | None = None, path_provider: PathProvider | None = None, data_container: DataContainer | None = None, static_context: dict[str, Any] | None = None):
                super().__init__(model_config, ...)

                self.submodel1 = MyModel1(
                    model_config=model_config.submodel1_config,
                    is_frozen=model_config.is_frozen,
                    update_counter=update_counter,
                    path_provider=path_provider,
                    data_container=data_container,
                    static_context=static_context,
                    optimizer_config=model_config.submodel1_config.optimizer_config,
                )
                self.submodel2 = MyModel2(model_config=model_config.submodel2_config, ... )

            @property
            def submodels(self) -> dict[str, Model]:
                return dict(
                    submodel1=self.submodel1,
                    submodel2=self.submodel2,
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # define forward pass here using self.submodel1 and self.submodel2
                x = self.submodel1(x)
                x = self.submodel2(x)
                return x
    """

    def __init__(
        self,
        model_config: ModelBaseConfig,
        update_counter: UpdateCounter | None = None,
        path_provider: PathProvider | None = None,
        data_container: DataContainer | None = None,
    ):
        """Base class for composite models, i.e. models that consist of multiple submodels of type Model."""
        # Use the first initializer from the list if available
        init_config = model_config.initializers if model_config.initializers else []
        super().__init__(
            model_config=model_config,
            update_counter=update_counter,
            path_provider=path_provider,
            data_container=data_container,
            initializer_config=init_config,  # type: ignore
        )

    def _validate_submodels(self) -> None:
        """Validate that all submodels are of type Model."""
        for name, submodel in self.submodels.items():
            if not isinstance(submodel, Model):
                raise TypeError(f"Submodel {name} is not of type Model, but {type(submodel)}")

    @property
    @abc.abstractmethod
    def submodels(self) -> dict[str, ModelBase]:
        """Returns the submodels of the composite model. This method must be implemented by the subclass, otherwise a NotImplementedError is raised."""
        raise NotImplementedError("submodels property must be implemented by subclass")

    def get_named_models(self) -> dict[str, ModelBase]:
        """Returns a dict of {model_name: model}, e.g., to log all learning rates of all models/submodels."""
        result = {}
        for name, submodel in self.submodels.items():
            if submodel is None:
                continue
            named_submodels = submodel.get_named_models()
            for key, value in named_submodels.items():
                result[f"{name}.{key}"] = value
        return result

    @property
    def device(self) -> torch.device:
        devices = [submodel.device for submodel in self.submodels.values() if submodel is not None]
        if not all(device == devices[0] for device in devices[1:]):
            raise RuntimeError("All submodels must be on the same device")
        return devices[0]

    def initialize_weights(self) -> Self:
        """Initialize the weights of the model, calling the initializer of all submodules."""
        self._validate_submodels()
        for submodel in self.submodels.values():
            if submodel is None:
                continue
            submodel.initialize_weights()
        return self

    def apply_initializers(self) -> Self:
        """Apply the initializers to the model, calling the initializer of all submodules."""
        self._validate_submodels()
        for submodel in self.submodels.values():
            if submodel is None:
                continue
            submodel.apply_initializers()
        for initializer in self.initializers:
            initializer.init_weights(self)
            initializer.init_optimizer(self)
        return self

    def initialize_optimizer(self) -> None:
        """Initialize the optimizer of the model."""
        for submodel in self.submodels.values():
            if submodel is None:
                continue
            submodel.initialize_optimizer()
        if self.is_frozen:
            self.logger.info(f"{self.name} has only frozen submodels -> put into eval mode")
            self.eval()

    def optimizer_step(self, grad_scaler: GradScaler | None) -> None:
        """Perform an optimization step, calling all submodules' optimization steps."""
        for submodel in self.submodels.values():
            if submodel is None:
                continue
            if isinstance(submodel, Model) and submodel.optimizer is None:
                continue
            submodel.optimizer_step(grad_scaler)

    def optimizer_schedule_step(self) -> None:
        """Perform the optimizer learning rate scheduler step, calling all submodules' scheduler steps."""
        for submodel in self.submodels.values():
            if submodel is None:
                continue
            if isinstance(submodel, Model) and (submodel.optimizer is None or submodel.is_frozen):
                continue
            submodel.optimizer_schedule_step()

    def optimizer_zero_grad(self, set_to_none: bool = True) -> None:
        """Zero the gradients of the optimizer, calling all submodules' zero_grad methods."""
        for submodel in self.submodels.values():
            if submodel is None:
                continue
            if isinstance(submodel, Model) and submodel.optimizer is None:
                continue
            submodel.optimizer_zero_grad(set_to_none)

    @property
    def is_frozen(self) -> bool:
        return all(m is None or m.is_frozen for m in self.submodels.values())

    @is_frozen.setter
    def is_frozen(self, value) -> None:
        for submodel in self.submodels.values():
            if submodel is None:
                continue
            submodel.is_frozen = value  # type: ignore

    def train(self, mode=True) -> Self:
        """Set the model to train or eval mode.

        Overwrites the nn.Module.train method to avoid setting the model to train mode if it is frozen
        and to call all submodules' train methods.

        Args:
            mode: If True, set the model to train mode. If False, set the model to eval mode.
        """
        for submodel in self.submodels.values():
            if submodel is None:
                continue
            submodel.train(mode=mode)
        # avoid setting mode to train if whole network is frozen
        if self.is_frozen and mode is True:
            return super().train(mode=False)
        return super().train(mode=mode)

    def to(self, device, *args, **kwargs) -> Self:  # type: ignore[override]
        """Performs Tensor dtype and/or device conversion, calling all submodules' to methods.

        Args:
            device: The desired device of the tensor. Can be a string (e.g. "cuda:0") or "cpu".
        """
        if isinstance(device, str):
            try:
                device = torch.device(device)
            except RuntimeError as e:
                self.logger.error(f"Invalid device: {device}")
                raise e
        if not isinstance(device, torch.device):
            raise TypeError(f"Expected torch.device, got {type(device)}")
        for submodel in self.submodels.values():
            if submodel is None:
                continue
            submodel.to(*args, **kwargs, device=device)  # type: ignore[call-overload]
        return super().to(*args, **kwargs, device=device)  # type: ignore
