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
    from noether.core.utils.training.counter import UpdateCounter
    from noether.data.container import DataContainer
    from noether.training.trainers import BaseTrainer


class CallbackBase:
    """Base class for callbacks that execute something before/after training.

    Allows overwriting `before_training` and `after_training`.

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

            # THIS IS INSIDE A CUSTOM CALLBACK

            # log only to experiment tracker, not stdout
            self.writer.add_scalar(key="classification_accuracy", value=0.2)
            # log to experiment tracker and stdout (as "0.24")
            self.writer.add_scalar(
                key="classification_accuracy",
                value=0.23623,
                logger=self.logger,
                format_str=".2f",
            )

    Note:
        As evaluations are pretty much always done in torch.no_grad() contexts, the hooks implemented by callbacks
        are always executed within a torch.no_grad() context.
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

    _logger: logging.Logger | None = None

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
            data_container: :class:`~noether.data.container.DataContainer` instance that provides access to all datasets.
            tracker: :class:`~noether.core.trackers.BaseTracker` instance to log metrics to stdout/disk/online platform.
            log_writer: :class:`~noether.core.writers.LogWriter` instance to log metrics to stdout/disk/online platform.
            checkpoint_writer: :class:`~noether.core.writers.CheckpointWriter` instance to save checkpoints during training.
            metric_property_provider: :class:`~noether.core.providers.MetricPropertyProvider` instance to access properties of metrics.
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

    @property
    def checkpoint_key(self) -> str:
        """Key used to identify this callback's state in checkpoints.

        Returns the callback's ``id`` if set, otherwise falls back to the class name.
        """
        return self.name if self.name is not None else type(self).__name__

    @staticmethod
    def validate_checkpoint_keys(callbacks: list[CallbackBase]) -> None:
        """Validate that all stateful callbacks have unique checkpoint keys.

        Should be called early (e.g. when callbacks are first assembled) so that
        duplicate-key errors surface immediately rather than hours into training
        when the first checkpoint is saved.

        Args:
            callbacks: list of callbacks to validate.

        Raises:
            ValueError: If two stateful callbacks produce the same checkpoint key.
        """
        seen: dict[str, CallbackBase] = {}
        for cb in callbacks:
            if cb.state_dict() is None:
                continue
            key = cb.checkpoint_key
            if key in seen:
                raise ValueError(
                    f"Two stateful callbacks share checkpoint key '{key}': {seen[key]} and {cb}. "
                    "Set a unique 'id' in each callback config to disambiguate."
                )
            seen[key] = cb

    @staticmethod
    def build_callback_state_dict(callbacks: list[CallbackBase]) -> dict[str, Any]:
        """Build a keyed dict of state dicts for all stateful callbacks.

        Args:
            callbacks: list of callbacks to save state for.

        Returns:
            Dict mapping checkpoint keys to state dicts (only stateful callbacks included).

        Raises:
            ValueError: If two stateful callbacks produce the same checkpoint key.
        """
        CallbackBase.validate_checkpoint_keys(callbacks)
        result: dict[str, Any] = {}
        for cb in callbacks:
            sd = cb.state_dict()
            if sd is None:
                continue
            result[cb.checkpoint_key] = sd
        return result

    @staticmethod
    def load_callback_state_dicts(
        callbacks: list[CallbackBase],
        checkpoint_data: dict[str, Any] | list[Any],
        logger: logging.Logger,
    ) -> None:
        """Load state dicts into callbacks, matching by key (dict) or position (legacy list).

        Modifies callbacks in-place via their ``load_state_dict`` method (analogous to
        ``torch.nn.Module.load_state_dict``).

        Args:
            callbacks: current callbacks to load state into (mutated in-place).
            checkpoint_data: either a dict keyed by checkpoint_key (new format) or a list (legacy format).
            logger: logger for warnings.
        """
        if isinstance(checkpoint_data, list):
            # Legacy format: positional matching of stateful callbacks
            stateful_from_checkpoint = [sd for sd in checkpoint_data if sd is not None]
            stateful_current = [cb for cb in callbacks if cb.state_dict() is not None]
            if len(stateful_from_checkpoint) != len(stateful_current):
                raise ValueError(
                    f"Number of stateful callbacks in checkpoint ({len(stateful_from_checkpoint)}) doesn't match "
                    f"number of stateful callbacks in current trainer ({len(stateful_current)})."
                )
            for cb, sd in zip(stateful_current, stateful_from_checkpoint, strict=True):
                # Mutates cb in-place
                cb.load_state_dict(sd)
            return

        # New format: match by key
        CallbackBase.validate_checkpoint_keys(callbacks)
        current_by_key = {cb.checkpoint_key: cb for cb in callbacks if cb.state_dict() is not None}

        matched_keys = set(checkpoint_data.keys()) & set(current_by_key.keys())
        unmatched_in_checkpoint = set(checkpoint_data.keys()) - matched_keys
        unmatched_in_current = set(current_by_key.keys()) - matched_keys

        if unmatched_in_checkpoint:
            logger.warning(
                f"Stateful callbacks in checkpoint not found in current trainer (skipped): {unmatched_in_checkpoint}"
            )
        if unmatched_in_current:
            logger.warning(
                f"Stateful callbacks in current trainer not found in checkpoint (not loaded): {unmatched_in_current}"
            )

        # Mutates each matched callback in-place
        for key in matched_keys:
            current_by_key[key].load_state_dict(checkpoint_data[key])

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
        unnecessarily wasteful to also store the state in the callbacks `state_dict`. Therefore,
        `resume_from_checkpoint` is called when resuming a run, which allows callbacks to load their state from any
        file that was stored on the disk.

        Args:
            resumption_paths: :class:`~noether.core.providers.path.PathProvider` instance to access paths from the checkpoint to resume from.
            model: model of the current training run.
        """

    @property
    def logger(self) -> logging.Logger:
        """Logger for logging to stdout."""
        if self._logger is None:
            self._logger = logging.getLogger(str(self))
        return self._logger

    def before_training(self, *, update_counter: UpdateCounter) -> None:
        """Hook called once before the training loop starts.

        This method is intended to be overridden by derived classes to perform initialization
        tasks before training begins. Common use cases include:

        * Initializing experiment tracking (e.g., logging hyperparameters)
        * Printing model summaries or architecture details
        * Initializing specific data structures or buffers needed during training
        * Performing sanity checks on the data or configuration

        Note:
            This method is executed within a ``torch.no_grad()`` context.

        Args:
            update_counter: :class:`~noether.core.utils.training.counter.UpdateCounter` instance to access current training progress.
        """

    def after_training(self, *, update_counter: UpdateCounter) -> None:
        """Hook called once after the training loop finishes.

        This method is intended to be overridden by derived classes to perform cleanup or
        final reporting tasks after training is complete. Common use cases include:

        * Performing a final evaluation on the test set
        * Saving final model weights or artifacts
        * Sending notifications (e.g., via Slack or email) about the completed run
        * Closing or finalizing experiment tracking sessions

        Note:
            This method is executed within a ``torch.no_grad()`` context.

        Args:
            update_counter: :class:`~noether.core.utils.training.counter.UpdateCounter` instance to access current training progress.
        """
