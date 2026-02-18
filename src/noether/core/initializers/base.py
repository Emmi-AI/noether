#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import abc
import logging

from noether.core.callbacks.base import CallbackBase
from noether.core.models.base import ModelBase
from noether.core.providers import PathProvider
from noether.core.utils.training.training_iteration import TrainingIteration


class InitializerBase(abc.ABC):
    def __init__(self, path_provider: PathProvider):
        """Base class for model initializers.

        Args:
            path_provider: PathProvider instance to access paths to load models from.
        """
        self.logger = logging.getLogger(type(self).__name__)
        self.path_provider = path_provider

    @abc.abstractmethod
    def init_weights(self, model: ModelBase) -> None:
        """Initialize the model weights from the checkpoint.

        Args:
            model: the model to load the weights into.
        """
        raise NotImplementedError("init_weights must be implemented by the child class")

    @abc.abstractmethod
    def init_optimizer(self, model: ModelBase) -> None:
        """Initialize the optimizer for the model.

        Args:
            model: a model to initialize the optimizer for. Assumes the model has an attribute optim.
        """
        raise NotImplementedError("init_optim must be implemented by the child class")

    def init_trainer(self, trainer) -> None:
        """Initialize the trainer from the checkpoint.

        By default, does nothing. Can be overridden by child classes.

        Args:
            trainer: the trainer to initialize.
        """
        return None

    def init_callbacks(self, callbacks: list[CallbackBase], model: ModelBase) -> None:
        """Initialize the callbacks from the checkpoint.

        By default, does nothing. Can be overridden by child classes.

        Args:
            callbacks: the list of callbacks to initialize.
            model: the model associated with the callbacks.
        """
        return None

    def start_checkpoint(self) -> TrainingIteration:
        """Get the start checkpoint for the model.

        By default , returns a TrainingIteration starting from zero.

        Returns:
            TrainingIteration: the start checkpoint for the model.
        """
        return TrainingIteration(epoch=0, update=0, sample=0)
