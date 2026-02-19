#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import math
import sys
from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from torch.utils.data import DistributedSampler, SequentialSampler
from tqdm import tqdm

from noether.core.callbacks.base import CallbackBase
from noether.core.distributed import all_gather_nograd, all_gather_nograd_clipped
from noether.core.distributed.config import is_distributed, is_rank0
from noether.core.models import ModelBase
from noether.core.providers import MetricPropertyProvider
from noether.core.schemas.callbacks import CallBackBaseConfig, PeriodicDataIteratorCallbackConfig
from noether.core.trackers import BaseTracker
from noether.core.utils.common import snake_type_name
from noether.core.utils.common.stopwatch import Stopwatch
from noether.core.utils.logging import NoopTqdm, tensor_like_to_string
from noether.core.utils.torch import move_items_to_device
from noether.core.utils.training.counter import UpdateCounter
from noether.core.utils.training.training_iteration import TrainingIteration
from noether.core.writers import CheckpointWriter, LogWriter
from noether.data.base.dataset import Dataset
from noether.data.samplers import SamplerIntervalConfig

if TYPE_CHECKING:
    from noether.data.container import DataContainer
    from noether.training.trainers import BaseTrainer


IntervalType = Literal["epoch", "update", "sample", "eval"]
"""Type alias for periodic callback interval types.

Defines the unit of training progress used to trigger periodic callbacks:

* "epoch": Callback is triggered based on completed epochs
* "update": Callback is triggered based on optimizer update steps
* "sample": Callback is triggered based on number of samples processed
* "eval": Callback is triggered independent of schedule for post-training evaluation
"""


class PeriodicCallback(CallbackBase):
    """Base class for callbacks that are invoked periodically during training.

    PeriodicCallback extends :class:`~noether.core.callbacks.base.CallbackBase` to support periodic execution based on
    training progress. Callbacks can be configured to run at regular intervals defined by epochs, updates (optimizer
    steps), or samples (data points processed). This class implements the infrastructure for periodic invocation while
    child classes define the actual behavior via the :meth:`periodic_callback` method.


    Interval Configuration:
        Callbacks can be configured to run periodically using one or more of:

        * ``every_n_epochs``: Execute callback every N epochs
        * ``every_n_updates``: Execute callback every N optimizer updates
        * ``every_n_samples``: Execute callback every N samples processed

    Tracking vs. Periodic Execution:
        The class provides two types of hooks:

        * Tracking hooks (:meth:`track_after_accumulation_step`, :meth:`track_after_update_step`): Called on every
          accumulation/update step to track metrics continuously (e.g., for running averages).
          I.e., if you want to log an exponential moving average of the loss every epoch,
          the logging is done in the periodic callback;
          however, the tracking of the loss values for computing the moving average is done in the tracking hook.
        * Periodic hook (:meth:`periodic_callback`): Called only when the configured interval is reached, typically for
          expensive operations like evaluation or checkpointing.

    Examples:
        Creating a custom periodic callback that logs metrics every 10 epochs:

        .. code-block:: python

            class CustomMetricCallback(PeriodicCallback):
                def periodic_callback(
                    self,
                    *,
                    interval_type: IntervalType,
                    update_counter: UpdateCounter,
                    **kwargs,
                ) -> None:
                    # This method is called every 10 epochs
                    metric_value = self.compute_expensive_metric()
                    self.writer.add_scalar(
                        key="custom_metric",
                        value=metric_value,
                        logger=self.logger,
                    )


            # Configure in YAML:
            # callbacks:
            #   - kind: path.to.CustomMetricCallback
            #     every_n_epochs: 10

        Tracking metrics at every update and logging periodically:

        .. code-block:: python

            class RunningAverageCallback(PeriodicCallback):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.loss_accumulator = []

                def track_after_update_step(self, *, update_counter: UpdateCounter, times: dict[str, float]) -> None:
                    # Track at every update
                    self.loss_accumulator.append(self.trainer.last_loss)

                def periodic_callback(
                    self,
                    *,
                    interval_type: IntervalType,
                    update_counter: UpdateCounter,
                    **kwargs,
                ) -> None:
                    # Log periodically
                    avg_loss = sum(self.loss_accumulator) / len(self.loss_accumulator)
                    self.writer.add_scalar("avg_loss", avg_loss, logger=self.logger)
                    self.loss_accumulator.clear()

    Attributes:
        every_n_epochs: If set, callback is invoked every N epochs.
        every_n_updates: If set, callback is invoked every N optimizer updates.
        every_n_samples: If set, callback is invoked every N samples processed.
        batch_size: Batch size used during training.

    """

    def __init__(
        self,
        callback_config: CallBackBaseConfig,
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
            callback_config: Configuration for the callback. See
                :class:`~noether.core.schemas.callbacks.CallBackBaseConfig`
                for available options.
            trainer: Trainer of the current run.
            model: Model of the current run.
            data_container: :class:`~noether.data.container.DataContainer` instance that provides access to all datasets.
            tracker: :class:`~noether.core.trackers.BaseTracker` instance to log metrics to stdout/disk/online platform.
            log_writer: :class:`~noether.core.writers.LogWriter` instance to log metrics.
            checkpoint_writer: :class:`~noether.core.writers.CheckpointWriter` instance to save checkpoints.
            metric_property_provider: :class:`~noether.core.providers.MetricPropertyProvider` instance to access properties of metrics.
            name: Name of the callback.
        """
        super().__init__(
            trainer, model, data_container, tracker, log_writer, checkpoint_writer, metric_property_provider, name
        )

        self.every_n_epochs = callback_config.every_n_epochs
        self.every_n_updates = callback_config.every_n_updates
        self.every_n_samples = callback_config.every_n_samples
        self.batch_size = callback_config.batch_size

        if not (type(self).after_update == PeriodicCallback.after_update):
            raise AssertionError("Children shouldn't override 'after_update'")
        if not (type(self).after_epoch == PeriodicCallback.after_epoch):
            raise AssertionError("Children shouldn't override 'after_epoch'")

    def __str__(self):
        return f"{type(self).__name__}({self.get_interval_string_verbose()})"

    def _should_invoke_after_epoch(self, training_iteration: TrainingIteration) -> bool:
        """Check after every epoch if the PeriodicCallback should be invoked.

        Args:
            training_iteration: :class:`~noether.core.utils.training.training_iteration.TrainingIteration` to check.

        Returns:
            True if the callback should be invoked, False otherwise.
        """
        return (
            self.every_n_epochs is not None
            and training_iteration.epoch is not None
            and training_iteration.epoch % self.every_n_epochs == 0
        )

    # TODO: rename to _should_invoke_after_update or something similar

    def _should_invoke_after_update(self, training_iteration: TrainingIteration) -> bool:
        """Check after every update if the PeriodicCallback should be invoked.

        Args:
            training_iteration: :class:`~noether.core.utils.training.training_iteration.TrainingIteration` to check.

        Returns:
            True if the callback should be invoked, False otherwise.
        """
        return (
            self.every_n_updates is not None
            and training_iteration.update is not None
            and training_iteration.update % self.every_n_updates == 0
        )

    def _should_invoke_after_sample(self, training_iteration: TrainingIteration, effective_batch_size: int) -> bool:
        """Check after every sample if the PeriodicCallback should be invoked.

        Args:
            training_iteration: :class:`~noether.core.utils.training.training_iteration.TrainingIteration` to check.
            effective_batch_size: Effective batch size to use for the check (required for checking how many samples
                have been processed since the last update).

        Returns:
            True if the callback should be invoked, False otherwise.
        """
        if self.every_n_samples is not None and training_iteration.sample is not None:
            last_update_samples = training_iteration.sample - effective_batch_size
            prev_invocation = int(last_update_samples / self.every_n_samples)
            cur_invocation = int(training_iteration.sample / self.every_n_samples)
            if cur_invocation > prev_invocation:
                return True
        return False

    def periodic_callback(
        self,
        *,
        interval_type: IntervalType,
        update_counter: UpdateCounter,
        **kwargs,
    ) -> None:
        """Hook called periodically based on the configured intervals.

        This method is the primary entry point for periodic actions in subclasses. It is
        triggered when any of the configured intervals (``every_n_epochs``, ``every_n_updates``,
        or ``every_n_samples``) are reached.

        Subclasses should override this method to implement periodic logic such as:

        * Calculating and logging expensive validation metrics
        * Saving specific model checkpoints or artifacts
        * Visualizing training progress (e.g., plotting samples)
        * Adjusting training hyperparameters or model state

        Note:
            This method is executed within a ``torch.no_grad()`` context.

        Args:
            interval_type: "epoch", "update", "sample" or "eval" indicating which interval triggered this callback.
            update_counter: :class:`~noether.core.utils.training.counter.UpdateCounter` instance providing details about
                the current training progress (epoch, update, sample counts).
            **kwargs: Additional keyword arguments passed from the triggering hook
                (e.g., from :meth:`after_epoch` or :meth:`after_update`).
        """

    def track_after_accumulation_step(
        self,
        *,
        update_counter: UpdateCounter,
        batch: Any,
        losses: dict[str, torch.Tensor],
        update_outputs: dict[str, torch.Tensor] | None,
        accumulation_steps: int,
        accumulation_step: int,
    ) -> None:
        """Hook called after each individual gradient accumulation step.

        This method is invoked for every batch processed during training, regardless of whether
        an optimizer update is performed in that step (i.e., when ``accumulation_steps > 1``).
        It is primarily used for tracking metrics that should be averaged or aggregated
        across accumulation steps.

        Common use cases include:

        * Logging per-batch losses for high-frequency monitoring
        * Accumulating statistics across batches before an optimizer update
        * Implementing custom logging that needs access to individual batch data

        Note:
            This method is generally intended to be called within a ``torch.no_grad()`` context
            by the trainer to ensure no gradients are tracked during logging operations.

        Args:
            update_counter: :class:`~noether.core.utils.training.counter.UpdateCounter` instance to access current training progress.
            batch: The current data batch processed in this accumulation step.
            losses: Dictionary of computed losses for the current batch.
            update_outputs: Optional dictionary of model outputs for the current batch.
            accumulation_steps: Total number of accumulation steps before an optimizer update.
            accumulation_step: The current accumulation step index (0-indexed).
        """

    @torch.no_grad()
    def track_after_update_step(self, *, update_counter: UpdateCounter, times: dict[str, float]) -> None:
        """Hook called after each optimizer update step.

        This method is invoked after a successful optimizer step and parameter update.
        It is typically used for tracking metrics that should be recorded once per update
        cycle, such as:

        * Latest loss values
        * Learning rates
        * Model parameter statistics (norms, etc.)
        * Training throughput and timing measurements

        Unlike :meth:`periodic_callback`, this hook is called on every update step, making it
        suitable for maintaining running averages or high-frequency telemetry.

        Note:
            This method is executed within a ``torch.no_grad()`` context.

        Args:
            update_counter: :class:`~noether.core.utils.training.counter.UpdateCounter` instance to access current training progress.
            times: Dictionary containing time measurements for various parts of the training
                step (e.g., 'data_time', 'forward_time', 'backward_time', 'update_time').
        """

    @torch.no_grad()
    def after_epoch(self, update_counter: UpdateCounter, **kwargs) -> None:
        """Invoked after every epoch to check if callback should be invoked.

        Applies ``torch.no_grad()`` context.

        Args:
            update_counter: :class:`~noether.core.utils.training.counter.UpdateCounter` instance to access current training progress.
            **kwargs: Additional keyword arguments.
        """
        if self._should_invoke_after_epoch(update_counter.cur_iteration):
            self.periodic_callback(interval_type="epoch", update_counter=update_counter, **kwargs)

    @torch.no_grad()
    def after_update(self, update_counter: UpdateCounter, **kwargs) -> None:
        """Invoked after every update to check if callback should be invoked.

        Applies ``torch.no_grad()`` context.

        Args:
            update_counter: :class:`~noether.core.utils.training.counter.UpdateCounter` instance to access current training progress.
            **kwargs: Additional keyword arguments.
        """
        if type(self).periodic_callback == PeriodicCallback.periodic_callback:
            return
        if self._should_invoke_after_update(update_counter.cur_iteration):
            self.periodic_callback(
                interval_type="update",
                update_counter=update_counter,
                **kwargs,
            )
        if self._should_invoke_after_sample(
            update_counter.cur_iteration,
            update_counter.effective_batch_size,
        ):
            self.periodic_callback(interval_type="sample", update_counter=update_counter, **kwargs)

    @torch.no_grad()
    def at_eval(self, update_counter: UpdateCounter, **kwargs) -> None:
        self.periodic_callback(interval_type="eval", update_counter=update_counter, **kwargs)

    def updates_till_next_invocation(self, update_counter: UpdateCounter) -> int:
        """Calculate how many updates remain until this callback is invoked.

        Args:
            update_counter: :class:`~noether.core.utils.training.counter.UpdateCounter` instance to access current training progress.

        Returns:
            Number of updates remaining until the next callback invocation.
        """
        updates_per_interval = self.updates_per_interval(update_counter)
        if update_counter.cur_iteration.update is None:
            return updates_per_interval
        return updates_per_interval - update_counter.cur_iteration.update % updates_per_interval

    def updates_per_interval(self, update_counter: UpdateCounter) -> int:
        """Calculate how many updates are from one invocation of this callback to the next.

        Args:
            update_counter: :class:`~noether.core.utils.training.counter.UpdateCounter` instance to access current training progress.

        Returns:
            Number of updates between callback invocations.
        """
        if self.every_n_epochs is not None:
            assert self.every_n_updates is None and self.every_n_samples is None
            return update_counter.updates_per_epoch * self.every_n_epochs
        if self.every_n_updates is not None:
            assert self.every_n_epochs is None and self.every_n_samples is None
            return self.every_n_updates
        if self.every_n_samples is not None:
            assert self.every_n_epochs is None and self.every_n_updates is None
            # NOTE: uneven every_n_samples not supported
            assert self.every_n_samples % update_counter.effective_batch_size == 0
            return int(self.every_n_samples / update_counter.effective_batch_size)
        raise RuntimeError("no interval defined for PeriodicCallback")

    def get_interval_string_verbose(self) -> str:
        """Return interval configuration as a verbose string.

        Returns:
            Interval as, e.g., "every_n_epochs=1" for epoch-based intervals.
        """
        results = []
        if self.every_n_epochs is not None:
            results.append(f"every_n_epochs={self.every_n_epochs}")
        if self.every_n_updates is not None:
            results.append(f"every_n_updates={self.every_n_updates}")
        if self.every_n_samples is not None:
            results.append(f"every_n_samples={self.every_n_samples}")
        return ",".join(results)

    def to_short_interval_string(self) -> str:
        """Return interval configuration as a short string.

        Returns:
            Interval as, e.g., "E1" if ``every_n_epochs=1`` for epoch-based intervals.
        """
        results = []
        if self.every_n_epochs is not None:
            results.append(f"E{self.every_n_epochs}")
        if self.every_n_updates is not None:
            results.append(f"U{self.every_n_updates}")
        if self.every_n_samples is not None:
            results.append(f"S{self.every_n_samples}")
        return "_".join(results)


class PeriodicDataIteratorCallback(PeriodicCallback, metaclass=ABCMeta):
    """Base class for callbacks that perform periodic iterations over a dataset.

    PeriodicDataIteratorCallback extends :class:`PeriodicCallback` to support evaluations or computations that require
    iterating over an entire dataset. This is commonly used for validation/test set evaluation, computing metrics on
    held-out data, or any operation that needs to process batches from a dataset at regular training intervals.

    The class integrates with the training data pipeline by registering samplers that control when and how data is
    loaded. It handles the complete iteration workflow: data loading, batch processing, result collation across
    distributed ranks, and final processing.

    Workflow:
        1. **Iteration** (:meth:`_iterate_over_dataset`): When the periodic interval is reached, iterate through the
           dataset in batches.
        2. **Process Data** (:meth:`process_data`): Process a single batch (e.g., run model inference) and return
           results.
        3. **Collation** (:meth:`_collate_result`): Aggregate results across all batches and distributed ranks.
        4. **Processing** (:meth:`process_results`): Compute final metrics or perform actions with the aggregated
           results.

    Key Features:
        * **Distributed Support**: Automatically handles distributed evaluation with proper gathering across ranks and
          padding removal.
        * **Flexible Collation**: Supports collating various result types (tensors, dicts of tensors, lists).
        * **Data Pipeline Integration**: Uses :class:`~noether.data.samplers.SamplerIntervalConfig` to integrate with
          the interleaved sampler for efficient data loading.
        * **Progress Tracking**: Provides progress bars and timing information for data loading.

    Template Methods to Override:
        Child classes must implement :meth:`process_data` and
        :meth:`process_results`:

        * :meth:`process_data`: Process a single batch (e.g., run model inference).
        * :meth:`process_results`: Process the aggregated results from all batches.

    Examples:
        Basic validation accuracy callback that evaluates on a test set every epoch:

        .. code-block:: python

            class AccuracyCallback(PeriodicDataIteratorCallback):
                def __init__(self, *args, dataset_key="test", **kwargs):
                    super().__init__(*args, **kwargs)
                    self.dataset_key = dataset_key

                def process_data(self, batch, *, trainer_model):
                    # Run inference on batch
                    x = batch["x"].to(trainer_model.device)
                    y_true = batch["class"].clone()
                    y_pred = trainer_model(x)
                    return {"predictions": y_pred, "labels": y_true}

                def process_results(self, results, *, interval_type, update_counter, **_):
                    # Compute accuracy from aggregated results
                    y_pred = results["predictions"]
                    y_true = results["labels"]
                    accuracy = (y_pred.argmax(dim=1) == y_true).float().mean()

                    self.writer.add_scalar(
                        key="test/accuracy",
                        value=accuracy.item(),
                        logger=self.logger,
                        format_str=".4f",
                    )


            # Configure in YAML:
            # callbacks:
            #   - kind: path.to.AccuracyCallback
            #     every_n_epochs: 1
            #     dataset_key: "test"

        Advanced example with multiple return values and custom collation:

        .. code-block:: python

            class DetailedEvaluationCallback(PeriodicDataIteratorCallback):
                def process_data(self, batch, *, trainer_model):
                    x = batch["x"].to(trainer_model.device)
                    y = batch["label"]

                    # Return multiple outputs as tuple
                    logits = trainer_model(x)
                    embeddings = trainer_model.get_embeddings(x)
                    return logits, embeddings, y

                def process_results(self, results, *, interval_type, update_counter, **_):
                    # results is a tuple: (all_logits, all_embeddings, all_labels)
                    logits, embeddings, labels = results

                    # Compute multiple metrics
                    accuracy = (logits.argmax(dim=1) == labels).float().mean()
                    mean_embedding_norm = embeddings.norm(dim=-1).mean()

                    self.writer.add_scalar("accuracy", accuracy.item())
                    self.writer.add_scalar("embedding_norm", mean_embedding_norm.item())



    Attributes:
        dataset_key: Key to identify the dataset to iterate over from ``self.data_container``. Automatically set from
            the callback config.
        sampler_config: Configuration for the sampler that controls dataset iteration. Automatically set when dataset is initialized.
        total_data_time: Cumulative time spent waiting for data loading across all periodic callbacks.

    Note:
        * The :meth:`process_data` method is called within a ``torch.no_grad()`` context automatically.
        * For distributed training, results are automatically gathered across all ranks with proper padding removal.
    """

    def __init__(
        self,
        callback_config: PeriodicDataIteratorCallbackConfig,
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
            callback_config: Configuration for the callback. See
                :class:`~noether.core.schemas.callbacks.PeriodicDataIteratorCallbackConfig`
                for available options.
            trainer: Trainer of the current run.
            model: Model of the current run.
            data_container: :class:`~noether.data.container.DataContainer` instance that provides access to all datasets.
            tracker: :class:`~noether.core.trackers.BaseTracker` instance to log metrics to stdout/disk/online platform.
            log_writer: :class:`~noether.core.writers.LogWriter` instance to log metrics.
            checkpoint_writer: :class:`~noether.core.writers.CheckpointWriter` instance to save checkpoints.
            metric_property_provider: :class:`~noether.core.providers.MetricPropertyProvider` instance to access properties of metrics.
            name: Name of the callback.
        """
        super().__init__(
            callback_config,
            trainer,
            model,
            data_container,
            tracker,
            log_writer,
            checkpoint_writer,
            metric_property_provider,
            name,
        )
        self.dataset_key = callback_config.dataset_key  # type: ignore
        self.total_data_time = 0.0
        self.sampler_config = self._sampler_config_from_key(key=self.dataset_key)

    def _sampler_config_from_key(
        self, key: str | None, properties: set[str] | None = None, max_size: int | None = None
    ) -> SamplerIntervalConfig:
        """Register the dataset that is used for this callback in the dataloading pipeline.

        Args:
            key: Key for identifying the dataset from ``self.data_container``. Uses the first dataset if ``None``.
            properties: Optionally specifies a subset of properties to load from the dataset.
            max_size: If provided, only uses a subset of the full dataset. Default: ``None`` (no subset).

        Returns:
            :class:`~noether.data.samplers.SamplerIntervalConfig` for the registered dataset.
        """
        dataset = self.data_container.get_dataset(key=key, properties=properties, max_size=max_size)
        config = self._create_sampler_config(dataset=dataset, pipeline=dataset.pipeline)  # type: ignore
        self.logger.info(f"{self} registered sampler {key} of {dataset} using {config.pipeline}")
        return config

    def _create_sampler_config(self, dataset: Dataset, pipeline=None) -> SamplerIntervalConfig:
        assert len(dataset) > 0
        return SamplerIntervalConfig(
            sampler=DistributedSampler(dataset, shuffle=False) if is_distributed() else SequentialSampler(dataset),
            every_n_epochs=self.every_n_epochs,
            every_n_updates=self.every_n_updates,
            every_n_samples=self.every_n_samples,
            pipeline=pipeline,
            batch_size=self.batch_size,
        )

    @abstractmethod
    def process_data(self, batch, *, trainer_model: torch.nn.Module) -> Any:
        """Template method that is called for each batch that is loaded from the dataset.

        This method should process a single batch and return results that will be collated.

        Args:
            batch: The loaded batch.
            trainer_model: Model of the current training run.

        Returns:
            Processed results for this batch. Can be a tensor, dict of tensors, list, or tuple.
        """
        ...

    def process_results(self, results: Any, *, interval_type, update_counter: UpdateCounter, **_) -> None:
        """Template method that is called with the collated results from dataset iteration.

        For example, metrics can be computed from the results for the entire test/validation dataset and logged.

        Args:
            results: The collated results that were produced by :meth:`_iterate_over_dataset` and the individual
                :meth:`process_data` calls.
            interval_type: The type of interval that triggered this callback invocation.
            update_counter: :class:`~noether.core.utils.training.counter.UpdateCounter` with the current training state.
            **_: Additional unused keyword arguments.
        """

    def _iterate_over_dataset(
        self,
        batch_size: int,
        data_iter: Iterator,
        trainer_model,
    ) -> Any:
        """Iterate over the registered dataset and collate results.

        For each loaded batch, :meth:`process_data` is called. The results are stored, postprocessed, collated and
        then returned. The postprocessing step ensures that padding for distributed evaluation is removed by gathering
        results across all ranks and cutting away padded entries.

        Args:
            batch_size: Batch size that is used for training. Used by default if ``self.batch_size`` is ``None``.
            data_iter: Iterator of the dataloading pipeline to fetch batches according to the registered
                sampler_configs.
            trainer_model: Model of the current training run.

        Returns:
            The collated results produced by iterating over the dataset, passing the samples through
            :meth:`process_data` and then collating the results (i.e., concatenating them and gathering them across
            ranks).

        Note:
            Collation is not implemented for arbitrary objects that :meth:`process_data` returns. It is suggested that
            :meth:`process_data` returns a dictionary of scalars.
        """
        if self.sampler_config is None:
            raise ValueError("Sampler config not registered.")
        config = self.sampler_config
        sampler: Any = config.sampler

        if isinstance(sampler, DistributedSampler):
            global_dataset_len = len(sampler.dataset)  # type: ignore
        else:
            global_dataset_len = len(sampler)
        local_dataset_len = len(sampler)
        num_batches = math.ceil(local_dataset_len / (config.batch_size or batch_size))

        # iterate
        data_times = []
        results = []
        pbar_ctor = NoopTqdm if not sys.stdout.isatty() or not is_rank0() else tqdm
        for _ in pbar_ctor(iterable=range(num_batches)):
            with Stopwatch() as data_sw:
                batch = next(data_iter)
                batch = move_items_to_device(self.trainer.device, batch)
            data_times.append(data_sw.elapsed_seconds)

            results.append(self.process_data(batch, trainer_model=trainer_model))

        mean_data_time = float(np.mean(data_times))
        self.logger.info(f"waited {mean_data_time:.2f}s for dataloading")
        self.total_data_time += mean_data_time

        single_output = False
        if not isinstance(results[0], tuple):
            results = [(res,) for res in results]
            single_output = True
        collated = [
            self._collate_result(result, global_dataset_len=global_dataset_len) for result in zip(*results, strict=True)
        ]

        if single_output:
            return collated[0]

        return collated

    def periodic_callback(  # type: ignore[override]
        self,
        *,
        interval_type: IntervalType,
        update_counter: UpdateCounter,
        data_iter: Iterator,
        trainer_model,
        batch_size: int,
        **_,
    ) -> None:
        results = self._iterate_over_dataset(
            batch_size=batch_size,
            data_iter=data_iter,
            trainer_model=trainer_model,
        )

        self.process_results(results, interval_type=interval_type, update_counter=update_counter)

    @staticmethod
    def _collate_tensors(tensors):
        if tensors[0].ndim == 0:
            return torch.stack(tensors)
        return torch.concat(tensors)

    @staticmethod
    def _collate_result(result, global_dataset_len):
        if isinstance(result[0], dict):
            # tuple[dict] -> dict[tensor]
            result = {
                k: PeriodicDataIteratorCallback._collate_tensors([r[k] for r in result]) for k in result[0].keys()
            }
            result = {k: all_gather_nograd_clipped(v, global_dataset_len) for k, v in result.items()}
        else:
            if isinstance(result[0], list):
                # List[List[Tensor]] -> List[Tensor]
                result = [torch.concat(item) for item in zip(*result, strict=True)]
                result = [all_gather_nograd_clipped(item, global_dataset_len) for item in result]
            elif result[0] is None:
                return None
            else:
                if torch.is_tensor(result[0]):
                    # List[Tensor] -> Tensor
                    if result[0].ndim == 0:
                        result = torch.stack(result)
                    else:
                        result = torch.concat(result)
                else:
                    result = torch.tensor(result)
                result = all_gather_nograd_clipped(result, global_dataset_len)
        return result

    @torch.no_grad()
    def after_training(self, **_) -> None:
        total_data_time = all_gather_nograd(self.total_data_time)
        self.logger.info(f"{snake_type_name(self)} total_data_time: {tensor_like_to_string(total_data_time)}")
