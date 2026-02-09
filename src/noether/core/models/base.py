#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import abc
import logging
from typing import TYPE_CHECKING, Self

import torch
from torch import nn

from noether.core.factory import Factory

if TYPE_CHECKING:  # import only for type checking to avoid circular imports
    from torch.amp.grad_scaler import GradScaler

    from noether.core.initializers import InitializerBase
    from noether.core.optimizer import OptimizerWrapper
    from noether.core.providers import PathProvider
    from noether.core.schemas.initializers import InitializerConfig
    from noether.core.schemas.models import ModelBaseConfig
    from noether.core.utils.training.counter import UpdateCounter
    from noether.data.container import DataContainer


class ModelBase(nn.Module):
    """Base class for all models trainable by the trainers."""

    def __init__(
        self,
        model_config: ModelBaseConfig,
        update_counter: UpdateCounter | None = None,
        path_provider: PathProvider | None = None,
        data_container: DataContainer | None = None,
        initializer_config: list[InitializerConfig] | None = None,
    ):
        """Base class for models that is used to define the interface for all models trainable by the trainers.

        Provides methods to initialize the model weights and setup (model-specific) optimizers.

        Args:
            model_config: Model configuration. See :class:`~noether.core.schemas.models.ModelBaseConfig` for available options.
            update_counter: The :class:`~noether.core.utils.training.counter.UpdateCounter` provided to the optimizer.
            path_provider: A path :class:`~noether.core.providers.path.PathProvider` used by the initializer to store or retrieve checkpoints.
            data_container: The :class:`~noether.data.container.DataContainer` which includes the data and dataloader.
                This is currently unused but helpful for quick prototyping only, evaluating forward in debug mode, etc.
            initializer_config: The initializer config used to initialize the model e.g. from a checkpoint.
        """
        super().__init__()
        self.logger = logging.getLogger(type(self).__name__)
        self.name = model_config.name
        self.update_counter = update_counter
        self.path_provider = path_provider
        self.data_container = data_container
        self._optim: OptimizerWrapper | None = None
        self.initializers: list[InitializerBase] = Factory().create_list(
            initializer_config,
            path_provider=self.path_provider,
        )
        self.model_config = model_config

        self.is_initialized = False

    @property
    def optimizer(self) -> OptimizerWrapper | None:
        return self._optim

    @property
    def device(self) -> torch.device:
        raise NotImplementedError

    @property
    def is_frozen(self) -> bool:
        raise NotImplementedError

    @property
    def param_count(self) -> int:
        """Returns the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_param_count(self) -> int:
        """Returns the number of parameters that require gradients (i.e., are trainable)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def frozen_param_count(self) -> int:
        """Returns the number of parameters that do not require gradients (i.e., are frozen)."""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    @property
    def nograd_paramnames(self) -> list[str]:
        """Returns a list of parameter names that do not have gradients (i.e., grad is None) but require gradients."""
        return [name for name, param in self.named_parameters() if param.grad is None and param.requires_grad]

    def initialize(self):
        """Initializes weights and optimizer parameters of the model."""
        self.initialize_weights()
        self.initialize_optimizer()
        self.apply_initializers()
        self.is_initialized = True
        return self

    @abc.abstractmethod
    def get_named_models(self) -> dict[str, ModelBase]:
        """Returns a dict of {model_name: model}, e.g., to log all learning rates of all models/submodels."""
        raise NotImplementedError("initialize_weights must be implemented by the subclass")

    @abc.abstractmethod
    def initialize_weights(self) -> Self:
        """Initialize the weights of the model."""
        raise NotImplementedError("initialize_weights must be implemented by the subclass")

    @abc.abstractmethod
    def apply_initializers(self) -> Self:
        """Apply the initializers to the model."""
        raise NotImplementedError("apply_initializers must be implemented by the subclass")

    @abc.abstractmethod
    def initialize_optimizer(self) -> None:
        """Initialize the optimizer of the model."""
        raise NotImplementedError("initialize_optim must be implemented by the subclass")

    @abc.abstractmethod
    def optimizer_step(self, grad_scaler: GradScaler | None) -> None:
        """Perform an optimization step."""
        raise NotImplementedError("optim_step must be implemented by the subclass")

    @abc.abstractmethod
    def optimizer_schedule_step(self) -> None:
        """Perform the optimizer learning rate scheduler step."""
        raise NotImplementedError("optim_schedule_step must be implemented by the subclass")

    @abc.abstractmethod
    def optimizer_zero_grad(self, set_to_none: bool = True) -> None:
        """Zero the gradients of the optimizer."""
        raise NotImplementedError("optim_zero_grad must be implemented by the subclass")
