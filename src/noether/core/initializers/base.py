#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import abc
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from noether.core.providers import PathProvider
from noether.core.utils.training.training_iteration import TrainingIteration

if TYPE_CHECKING:
    from noether.core.callbacks.base import CallbackBase
    from noether.core.models.base import ModelBase


class InitializerConfig(BaseModel):
    kind: str = Field(default="noether.core.initializers.PreviousRunInitializer")
    kwargs: dict[str, Any] | None = None
    """Additional keyword arguments to pass to the initializer."""
    run_id: str
    """A unique identifier for the training stage. This is used to find the correct checkpoint."""

    stage_name: str | None = None
    """The name of the stage training stage if defined. When training, the stage name is usually "train"."""
    model_name: str | None = None
    """The name of the model to load. This is the model_name used in CheckpointCallback."""
    checkpoint_tag: str | None | dict = None
    """Which checkpoint to load.
    Checkpoint is usually "latest" or "best_loss", or "E*_U*_S*", depending on which checkpoint you want to load.
    """
    model_info: str | None = None
    """Optional string to provide additional info about the model weights in the checkpoint filename. E.g., the stored weights are the EMA, or in a different precision."""
    model_config = {"extra": "forbid"}


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
