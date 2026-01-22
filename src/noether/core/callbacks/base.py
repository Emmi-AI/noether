#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from noether.core.providers.metric_property import MetricPropertyProvider
from noether.core.providers.path import PathProvider
from noether.core.trackers import BaseTracker
from noether.core.writers import CheckpointWriter, LogWriter

if TYPE_CHECKING:
    from noether.core.models import ModelBase
    from noether.core.utils.training.counter import UpdateCounter  # fixme?
    from noether.data.container import DataContainer
    from noether.training.trainers import BaseTrainer


class CallbackBase:
    """Base class for callbacks that log something before/after training.

    Allows overwriting `_before_training` and `_after_training`.

    If the callback is stateful (i.e., it tracks something across the training process that needs to be loaded if the
    run is resumed), there are two ways to implement loading the callback state:

    * `state_dict`: write current state into a state dict. When the trainer saves the current checkpoint to the disk,
      it will also store the `state_dict` of all callbacks within the trainer `state_dict`. Once a run is resumed, a
      callback can load its state from the previously stored `state_dict` by overwriting the `load_state_dict`.

    * `resume_from_checkpoint`: If a callback is storing large files onto the disk, it would be redudant to also
      store them within its `state_dict`. Therefore, this method is called on resume to allow callbacks to load their
      state from files on the disk.

    Callbacks have access to a `LogWriter`, with which callbacks can log metrics. The `LogWriter` is a singleton.

    Examples:
        .. code-block:: python

            # log only to experiment tracker, not stdout
            self.writer.add_scalar(key="classification_accuracy", value=0.2)
            # log to experiment tracker and stdout (as "0.24")
            self.writer.add_scalar(
                key="classification_accuracy",
                value=0.23623,
                logger=self.logger,
                format_str=".2f",
            )

    As evaluations are pretty much always done in torch.no_grad() contexts, the hooks implemented by callbacks
    automatically apply the torch.no_grad() context. Therefore, the `CallbackBase` class makes use of the "template
    method" design pattern, where templates (e.g. `before_training`) implement the invariant behavior (e.g., applying
    torch.no_grad()). The template implementations start with an underscore (e.g., `_before_training`) and only these
    methods should be implemented by child classes. Templates (e.g., `before_training`) should not be overwritten.
    """

    trainer: BaseTrainer
    """Trainer of the current run. Can be used to access training state."""
    model: ModelBase
    """Model of the current run. Can be used to access model parameters."""
    data_container: DataContainer
    """Data container of the current run. Can be used to access all datasets."""
    tracker: BaseTracker
    """Tracker of the current run. Can be used for direct access to the experiment tracking platform."""
    writer: LogWriter
    """Log writer of the current run. Can be used to log metrics to stdout/disk/online platform."""
    metric_property_provider: MetricPropertyProvider
    """Metric property provider of the current run. Defines properties of metrics (e.g., whether higher values are better)."""
    checkpoint_writer: CheckpointWriter
    """Checkpoint writer of the current run. Can be used to store checkpoints during training."""

    def __init__(
        self,
        trainer: BaseTrainer,
        model: ModelBase,
        data_container: DataContainer,
        tracker: BaseTracker,
        log_writer: LogWriter,
        checkpoint_writer: CheckpointWriter,
        metric_property_provider: MetricPropertyProvider,
        name: str | None = None,
    ):
        """

        Args:
            trainer: Trainer of the current run.
            model: Model of the current run.
            data_container: DataContainer instance that provides access to all datasets.
            tracker: Tracker instance to log metrics to stdout/disk/online platform.
            log_writer: LogWriter instance to log metrics to stdout/disk/online platform.
            metric_property_provider: MetricPropertyProvider instance to access properties of metrics.
            name: Name of the callback.

        """
        self.name = name
        self.trainer = trainer
        self.model = model
        self.data_container = data_container
        self.tracker = tracker
        self.writer = log_writer
        self.metric_property_provider = metric_property_provider
        self.checkpoint_writer = checkpoint_writer

        # these things are initialized on property access because they require the name/full_name
        # (which can be set from child classes)
        self._logger = None

        # check that children only override their implementation methods
        assert type(self).before_training == CallbackBase.before_training
        assert type(self).after_training == CallbackBase.after_training

    def __repr__(self):
        return str(self)

    def __str__(self):
        return type(self).__name__

    def state_dict(self) -> dict[str, torch.Tensor] | None:
        """If a callback is stateful, the state will be stored when a checkpoint is stored to the disk.

        Returns:
            State of the callback. By default, callbacks are non-stateful and return None.
        """
        return None

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """If a callback is stateful, the state will be stored when a checkpoint is stored to the disk and can be
        loaded with this method upon resuming a run.

        Args:
            state_dict: State to be loaded. By default, callbacks are non-stateful and load_state_dict does nothing.
        """

    def resume_from_checkpoint(self, resumption_paths: PathProvider, model: ModelBase) -> None:
        """If a callback stores large files to disk and is stateful (e.g., an EMA of the model), it would be
        unecessarily wasteful to also store the state in the callbacks `state_dict`. Therefore,
        `resume_from_checkpoint` is called when resuming a run, which allows callbacks to load their state from any
        file that was stored on the disk.

        Args:
            resumption_paths: PathProvider instance to access paths from the checkpoint to resume from.
            model: model of the current training run.
        """

    @property
    def logger(self) -> logging.Logger:
        """Logger for logging to stdout."""
        if self._logger is None:
            self._logger = logging.getLogger(str(self))
        return self._logger

    @torch.no_grad()
    def before_training(self, update_counter: UpdateCounter) -> None:
        """Hook that is called before training starts. Applies `torch.no_grad()` context.

        Args:
            update_counter: UpdateCounter instance to access current training progress.
        """
        self._before_training(update_counter=update_counter)

    @torch.no_grad()
    def after_training(self, update_counter: UpdateCounter) -> None:
        """Hook that is called after training ends. Applies `torch.no_grad()` context.

        Args:
            update_counter: UpdateCounter instance to access current training progress.
        """
        self._after_training(update_counter=update_counter)

    def _before_training(self, *, update_counter: UpdateCounter) -> None:
        """Template method for `before_training`. Derived classes should overwrite this method to do something before
        training starts. Implementing this method is optional.

        Args:
            update_counter: UpdateCounter instance to access current training progress.
        """

    def _after_training(self, *, update_counter: UpdateCounter) -> None:
        """Template method for `after_training`. Derived classes should overwrite this method to do something after
        training ends. Implementing this method is optional.

        Args:
            update_counter: UpdateCounter instance to access current training progress.
        """
