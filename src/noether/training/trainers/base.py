#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import logging
import os
import sys
import warnings
from collections import defaultdict
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch
import torch.utils.data
from torch import Tensor
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel

from noether.core.callbacks import CallbackBase, PeriodicCallback
from noether.core.callbacks.early_stoppers import EarlyStopIteration
from noether.core.callbacks.periodic import PeriodicDataIteratorCallback
from noether.core.distributed import (
    all_gather_nograd,
    get_num_nodes,
    get_world_size,
    is_distributed,
    is_rank0,
)
from noether.core.factory import Factory
from noether.core.initializers import InitializerBase
from noether.core.providers import (
    MetricPropertyProvider,
    PathProvider,
)
from noether.core.schemas import BaseTrainerConfig
from noether.core.schemas.callbacks import CallBackBaseConfig, OnlineLossCallbackConfig
from noether.core.trackers import BaseTracker
from noether.core.types import CheckpointKeys
from noether.core.utils.common.stopwatch import Stopwatch
from noether.core.utils.torch import get_grad_scaler_and_autocast_context, get_supported_precision, move_items_to_device
from noether.core.utils.training import TrainingIteration, UpdateCounter
from noether.core.writers import CheckpointWriter, LogWriter
from noether.training.trainers.constants import TRAINING_DATA_WAIT_TIME, TRAINING_UPDATE_TIME
from noether.training.trainers.types import LossResult, TrainerResult

if TYPE_CHECKING:  # import only for type checking to avoid circular imports
    from noether.core.models import ModelBase
    from noether.data.container import DataContainer


class TrainingContextFilter(logging.Filter):
    def __init__(self, update_counter: UpdateCounter):
        super().__init__()
        self.update_counter = update_counter

    def filter(self, record: logging.LogRecord) -> bool:
        if self.update_counter.cur_iteration:
            record.epoch = self.update_counter.cur_iteration.epoch
            record.max_epoch = self.update_counter.end_iteration.epoch
            record.update = self.update_counter.cur_iteration.update
            record.max_update = self.update_counter.end_iteration.update
        return True


class BaseTrainer:
    """Base class for all trainers that use SGD-based optimizers.

    This class implements the main training loop and provides utility functions for logging, checkpointing, and callbacks.
    In your down-stream you have to implement the `loss_compute` method that calculates the loss based on the model output and the targets.
    Optionally, you can also override the `train_step` method if you want to implement a custom training step (e.g., for multi-loss training or custom backward logic).
    If you only want to implement a custom loss calculation but keep the rest of the training loop, you can just override the `loss_compute` method.
    For example:

    .. code-block:: python

        class MyTrainer(BaseTrainer):
            def __init__(self, trainer_config: BaseTrainerConfig, **kwargs):
                super().__init__(trainer_config, **kwargs)

            def loss_compute(
                self, forward_output: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
            ) -> LossResult:
                # compute loss based on model output and targets
                return loss


    """

    def __init__(
        self,
        config: BaseTrainerConfig,
        data_container: DataContainer,
        device: str,
        tracker: BaseTracker,
        path_provider: PathProvider,
        main_sampler_kwargs: dict | None = None,
        metric_property_provider: MetricPropertyProvider | None = None,
    ):
        """

        Args:
            config: Configuration for the trainer. See :class:`~noether.core.schemas.BaseTrainerConfig` for the available options.
            data_container: The :class:`~noether.data.container.DataContainer` which includes the data and dataloader.
            device: The device to use for training (e.g., "cuda"). It is assumed that the process was configured such
                that only 1 device is visible (e.g., via the `CUDA_VISIBLE_DEVICES` environment variable).
            main_sampler_kwargs: Kwargs passed to instantiate the main sampler.
            tracker: The tracker to use for training.
            path_provider: The :class:`~noether.core.providers.PathProvider` to use for training.
            metric_property_provider: The :class:`~noether.core.providers.MetricPropertyProvider` to use for training.
        """
        self.logger = logging.getLogger(type(self).__name__)

        self.config = config
        self.data_container = data_container
        self.path_provider = path_provider
        self.main_sampler_kwargs = main_sampler_kwargs

        self.device: torch.device = torch.device(device)
        self.end_checkpoint = TrainingIteration(config.max_epochs, config.max_updates, config.max_samples)
        self.precision = get_supported_precision(
            desired_precision=config.precision,
            device=self.device,
        )
        self.logger.info(f"using precision: {self.precision} (desired={config.precision})")
        self.grad_scaler, self.autocast_context = get_grad_scaler_and_autocast_context(self.precision, self.device)

        eff_len = len(self.data_container.get_dataset("train"))
        if eff_len < self.config.effective_batch_size:
            raise ValueError(
                f"Effective dataset length {eff_len} is less than the configured effective batch size {self.config.effective_batch_size}"
            )

        self.updates_per_epoch = int(eff_len / config.effective_batch_size)
        self.skip_nan_loss_counter = 0

        self.initializer: InitializerBase | None = Factory().create(
            config.initializer,
            path_provider=self.path_provider,
        )

        if self.initializer is not None and not isinstance(self.initializer, InitializerBase):
            raise TypeError("initializer must be of type InitializerBase")

        if self.initializer is None:
            if config.start_at_epoch is not None:
                start_epoch = config.start_at_epoch
                start_update = self.updates_per_epoch * start_epoch
                start_sample = start_update * config.effective_batch_size
            else:
                start_epoch = 0
                start_update = 0
                start_sample = 0
            self.start_checkpoint = TrainingIteration(epoch=start_epoch, update=start_update, sample=start_sample)
        else:
            if config.start_at_epoch is not None:
                raise ValueError(
                    "cannot use both resume initializer and start_at_epoch, because start epoch is stored in the checkpoint"
                )
            self.start_checkpoint = self.initializer.start_checkpoint()

            if not (
                self.start_checkpoint.epoch is not None
                and self.start_checkpoint.epoch * self.updates_per_epoch == self.start_checkpoint.update
            ):
                raise ValueError(
                    "resuming from non-epoch based checkpoint is not supported (DataLoading is tricky in this setting)"
                )

        self.tracker = tracker
        self.path_provider = path_provider

        self.metric_property_provider = metric_property_provider

        self.update_counter = UpdateCounter(
            start_iteration=self.start_checkpoint,
            end_iteration=self.end_checkpoint,
            updates_per_epoch=self.updates_per_epoch,
            effective_batch_size=config.effective_batch_size,
        )

        self.log_writer = LogWriter(
            path_provider=self.path_provider,
            update_counter=self.update_counter,
            tracker=self.tracker,
        )

        self.checkpoint_writer = CheckpointWriter(path_provider=self.path_provider, update_counter=self.update_counter)

        self.callbacks: list[CallbackBase] = []

        # check that children only override their implementation methods
        if not type(self).train == BaseTrainer.train:
            raise ValueError("Derived classes should not implement the train method.")
        if not type(self).wrap_model == BaseTrainer.wrap_model:
            raise ValueError("Derived classes should not implement the wrap_model method.")

        self._has_logged_unused_params = False
        self._skip_nan_step = False

        self.forward_properties = config.forward_properties if config.forward_properties is not None else []
        self.target_properties = config.target_properties if config.target_properties is not None else []

        self.batch_keys = set(self.forward_properties).union(set(self.target_properties))

    def get_user_callbacks(self, model: ModelBase, evaluation=False) -> list[CallbackBase]:
        callback_default_args = self._get_default_callback_kwargs(model)
        callbacks: list[CallbackBase] = Factory().create_list(self.config.callbacks, **callback_default_args)
        return callbacks

    def get_all_callbacks(self, model: ModelBase) -> list[CallbackBase]:
        """Get all callbacks including default/trainer callbacks."""
        callback_default_args = self._get_default_callback_kwargs(model)

        callbacks = self.get_user_callbacks(model)
        if self.config.add_default_callbacks:
            callbacks += self.get_default_callbacks(callback_default_args)
        if self.config.add_trainer_callbacks:
            callbacks += self.get_trainer_callbacks(callback_default_args)
        return callbacks

    def get_trainer_callbacks(self, callback_default_args: dict[str, Any]) -> list[CallbackBase]:
        """Get trainer-specific callbacks. This may optionally be overridden by derived classes."""
        return []

    def _get_default_callback_kwargs(self, model: ModelBase) -> dict[str, Any]:
        """Get default kwargs for callbacks constructor."""

        return dict(
            data_container=self.data_container,
            trainer=self,
            model=model,
            tracker=self.tracker,
            log_writer=self.log_writer,
            checkpoint_writer=self.checkpoint_writer,
            metric_property_provider=self.metric_property_provider,
        )

    def get_default_callback_intervals(self) -> dict[str, Any]:
        """Get default intervals at which callbacks are called."""
        return dict(
            every_n_epochs=self.config.log_every_n_epochs,
            every_n_updates=self.config.log_every_n_updates,
            every_n_samples=self.config.log_every_n_samples,
        )

    def get_default_callbacks(self, default_kwargs: dict[str, Any]) -> list[CallbackBase]:
        # Local import to avoid circular dependencies
        from noether.core.callbacks import DatasetStatsCallback, OnlineLossCallback, ParamCountCallback

        """Get default callbacks."""
        # statistic callbacks
        default_callbacks: list[CallbackBase] = [
            DatasetStatsCallback(**default_kwargs),
            ParamCountCallback(**default_kwargs),
        ]

        default_intervals = self.get_default_callback_intervals()

        # add default training loggers (not needed for eval runs)
        if not self.update_counter.is_finished:
            from noether.core.callbacks import (
                EtaCallback,
                PeakMemoryCallback,
                ProgressCallback,
                TrainTimeCallback,
            )

            if all(v is None for v in default_intervals.values()):
                default_intervals["every_n_updates"] = max(self.total_training_updates // 100, 1)
                self.logger.warning(
                    f"No logging intervals set, defaulting to every {default_intervals['every_n_updates']} updates for logging callbacks."
                )
            default_callbacks += [
                ProgressCallback(
                    callback_config=CallBackBaseConfig.model_validate(default_intervals), **default_kwargs
                ),
                TrainTimeCallback(
                    callback_config=CallBackBaseConfig.model_validate(default_intervals), **default_kwargs
                ),
                PeakMemoryCallback(
                    callback_config=CallBackBaseConfig.model_validate(default_intervals), **default_kwargs
                ),
                OnlineLossCallback(
                    callback_config=OnlineLossCallbackConfig.model_validate({**default_intervals, "verbose": True}),
                    **default_kwargs,
                ),
            ]

            # EtaCallback is pointless in non-interactive/non-tty settings
            if sys.stdout.isatty() and is_rank0():
                default_callbacks.append(
                    EtaCallback(callback_config=CallBackBaseConfig.model_validate(default_intervals), **default_kwargs)
                )

            track_config = dict(
                every_n_epochs=self.config.track_every_n_epochs,
                every_n_updates=self.config.track_every_n_updates,
                every_n_samples=self.config.track_every_n_samples,
            )
            if all(v is None for v in track_config.values()):
                track_config["every_n_updates"] = max(self.total_training_updates // 500, 1)
                self.logger.warning(
                    f"No tracking intervals set, defaulting to every {track_config['every_n_updates']} updates for tracking callbacks."
                )

            from noether.core.callbacks import LrCallback

            default_callbacks += [
                LrCallback(callback_config=CallBackBaseConfig.model_validate(track_config), **default_kwargs),
                OnlineLossCallback(
                    callback_config=OnlineLossCallbackConfig.model_validate({**track_config, "verbose": False}),
                    **default_kwargs,
                ),
            ]

        for callback in default_callbacks:
            self.logger.debug(f"added default {callback}")
        return default_callbacks

    def _calculate_batch_size_and_accumulation_steps(self):
        world_size = get_world_size()
        if not self.config.effective_batch_size % world_size == 0:
            raise ValueError(
                f"effective_batch_size ({self.config.effective_batch_size}) needs to be multiple of world_size ({world_size})"
            )
        effective_batch_size_per_device = int(self.config.effective_batch_size / world_size)
        if self.end_checkpoint.update == 0:
            self.logger.info("eval run -> no automatic batchsize")
            return effective_batch_size_per_device, 1
        if self.config.disable_gradient_accumulation:
            self.logger.debug("gradient accumulation disabled")
            return effective_batch_size_per_device, 1
        if get_num_nodes() > 1 and self.config.max_batch_size is None:
            self.logger.info("found multi-node setting -> disable automatic batchsize (occasionally hangs)")
            return effective_batch_size_per_device, 1
        if self.config.use_torch_compile and self.config.max_batch_size is None:
            self.logger.info("torch.compile is used -> automatic batchsize not supported")
            return effective_batch_size_per_device, 1

        if is_distributed():
            self.logger.debug(f"effective_batch_size_per_device: {effective_batch_size_per_device}")
            self.logger.debug(f"world_size: {get_world_size()}")

        if self.config.max_batch_size is None:
            raise ValueError("gradient accumulation requires max_batch_size to be set")
        if not self.config.max_batch_size % world_size == 0:
            raise ValueError(
                f"max_batch_size ({self.config.max_batch_size}) needs to be multiple of world_size ({world_size})"
            )

        max_batch_size = int(self.config.max_batch_size / world_size)
        self.logger.info(f"Using provided max_batch_size {self.config.max_batch_size} ({max_batch_size} per device)")

        # calculate batch_size and accumulation_steps
        if effective_batch_size_per_device <= max_batch_size:
            # fits into memory
            batch_size = effective_batch_size_per_device
            accumulation_steps = 1
        else:
            # multiple accumulation steps
            if not effective_batch_size_per_device % max_batch_size == 0:
                raise ValueError("effective_batch_size_per_device needs to be multiple of max_batch_size")
            accumulation_steps = int(effective_batch_size_per_device / max_batch_size)

            batch_size = int(effective_batch_size_per_device / accumulation_steps)
        return batch_size, accumulation_steps

    def state_dict(self) -> dict[str, Any]:
        """Get the state dict of the trainer."""
        callback_state_dicts = [callback.state_dict() for callback in self.callbacks]
        state_dict: dict[str, Any] = {
            CheckpointKeys.CALLBACK_STATE_DICT: callback_state_dicts,
            CheckpointKeys.TRAINING_ITERATION: dict(self.update_counter.cur_iteration),
        }
        if isinstance(self.grad_scaler, GradScaler):
            state_dict[CheckpointKeys.GRAD_SCALER] = self.grad_scaler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state dict of the trainer."""
        # shallow copy
        state_dict = dict(state_dict.items())

        # load callback state_dicts
        callback_state_dicts = state_dict.pop(CheckpointKeys.CALLBACK_STATE_DICT)

        if len(callback_state_dicts) != len(self.callbacks):
            raise ValueError(
                f"Number of callbacks in checkpoint ({len(callback_state_dicts)}) does not match number of current callbacks ({len(self.callbacks)})"
            )
        for callback, sd in zip(self.callbacks, callback_state_dicts, strict=True):
            callback.load_state_dict(sd)

        # load grad_scaler
        grad_scaler_state_dict = state_dict.pop(CheckpointKeys.GRAD_SCALER, None)
        if isinstance(self.grad_scaler, GradScaler):
            if grad_scaler_state_dict := state_dict.pop(CheckpointKeys.GRAD_SCALER, None):
                self.grad_scaler.load_state_dict(grad_scaler_state_dict)
            else:
                self.logger.warning(
                    f"trainer checkpoint has no grad_scaler but current trainer uses {self.precision} precision"
                )

    def _prepare_model(self, model: ModelBase) -> ModelBase:
        model = model.to(self.device)
        model.initialize()
        self.apply_resume_initializer(model)
        return model

    def apply_resume_initializer(self, model: ModelBase) -> None:
        """Apply the resume initializer to the model."""
        # initialize model to state where it was resumed from
        if self.initializer is not None:
            self.initializer.init_trainer(self)
            self.initializer.init_weights(model)
            self.initializer.init_optimizer(model)
            self.initializer.init_callbacks(self.callbacks, model=model)

    def get_data_loader(
        self, iterator_callbacks: list[PeriodicDataIteratorCallback], batch_size: int, evaluation: bool = False
    ) -> torch.utils.data.DataLoader:
        """Get the data loader for training."""
        configs = []
        for c in iterator_callbacks:
            cur_config = c.sampler_config
            configs.append(cur_config)
        kwargs = {}
        if self.start_checkpoint.epoch != 0:
            kwargs["start_epoch"] = self.start_checkpoint.epoch
        train_collator = None
        if not evaluation:
            train_dataset = self.data_container.get_dataset("train")
            main_sampler = self.data_container.get_main_sampler(
                train_dataset=train_dataset,
                **(self.main_sampler_kwargs or {}),
            )
            if train_dataset.pipeline is None:
                raise ValueError("Pipeline is None for training dataset, which cannot be None for training.")
            train_collator = train_dataset.pipeline
        else:
            main_sampler = torch.utils.data.SequentialSampler(list())

        return self.data_container.get_data_loader(
            train_sampler=main_sampler,
            train_collator=train_collator,
            batch_size=batch_size,
            epochs=self.end_checkpoint.epoch,
            updates=self.end_checkpoint.update,
            samples=self.end_checkpoint.sample,
            callback_samplers=configs,
            evaluation=evaluation,
            **kwargs,
        )

    def _split_batch(self, batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Splits the input batch into forward inputs and targets based on the configured properties.

        Args:
            batch: The input batch containing all data.
        """
        if batch.keys() ^ self.batch_keys:
            missing_keys = self.batch_keys - batch.keys()
            additional_keys = batch.keys() - self.batch_keys
            warnings.warn(f"Batch contains additional keys {additional_keys} or is missing keys: {missing_keys}")

        forward_batch = {k: v for k, v in batch.items() if k in self.forward_properties}
        targets_batch = {k: v for k, v in batch.items() if k in self.target_properties}

        return forward_batch, targets_batch

    def loss_compute(
        self, forward_output: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> LossResult | tuple[LossResult, dict[str, torch.Tensor]]:
        """
        Each trainer that extends this class needs to implement a custom loss computation using the targets and the model output.

        Args:
            forward_output: Output of the model after the forward pass.
            targets: Dict with target tensors needed to compute the loss for this trainer.

        Returns:
            A dict with the (weighted) sub-losses to log. Or a tuple of (losses, additional_outputs) where additional_outputs is
            a dict with additional information about the model forward pass that is passed to the track_after_accumulation_step method of the callbacks, e.g., the logits and targets to calculate a training accuracy in a callback).

        Note: If a tuple is returned, the second element will be passed as additional_outputs in the TrainerResult returned by the train_step method.
        """
        raise NotImplementedError("Subclasses must implement loss_compute.")

    def train_step(self, batch: dict[str, Tensor], model: torch.nn.Module) -> TrainerResult:
        """Overriding this function is optional. By default, the `train_step` of the model will be called and is
        expected to return a TrainerResult. Trainers can override this method to implement custom training logic.

        Args:
            batch: Batch of data from which the loss is calculated.
            model: Model to use for processing the data.

        Returns:
            TrainerResult dataclass with the loss for backpropagation, (optionally) individual losses if multiple
            losses are used, and (optionally) additional information about the model forward pass that is passed
            to the callbacks (e.g., the logits and targets to calculate a training accuracy in a callback).
        """
        forward_batch, targets_batch = self._split_batch(batch)
        forward_output = model(**forward_batch)
        additional_outputs = None
        losses = self.loss_compute(forward_output=forward_output, targets=targets_batch)

        if isinstance(losses, tuple) and len(losses) == 2:
            losses, additional_outputs = losses

        if isinstance(losses, torch.Tensor):
            return TrainerResult(
                total_loss=losses, additional_outputs=additional_outputs, losses_to_log={"loss": losses}
            )
        elif isinstance(losses, list):
            losses = {f"loss_{i}": loss for i, loss in enumerate(losses)}

        if len(losses) == 0:
            raise ValueError("No losses computed, check your output keys and loss function.")

        return TrainerResult(
            total_loss=sum(losses.values(), start=torch.zeros_like(next(iter(losses.values())))),
            losses_to_log=losses,
            additional_outputs=additional_outputs,
        )

    def wrap_model(self, model: ModelBase) -> torch.nn.Module:
        """Wrap the model for training, return the model, wrapped model and ddp+compiled model."""
        if not model.is_initialized:
            raise ValueError("Model needs to be initialized before wrapping")
        ddp_model = self.wrap_ddp(model)
        return self.wrap_compile(ddp_model)

    def wrap_ddp(self, model: ModelBase) -> ModelBase | DistributedDataParallel:
        """Wrap the model with DistributedDataParallel in multi-GPU settings."""

        # DDP not needed if training on 1 GPU or CPU
        if not is_distributed() or model.device == torch.device("cpu"):
            return model

        trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable_param_count > 0:
            if self.config.find_unused_params:
                self.logger.info("using DDP find_unused_params")
            if self.config.static_graph:
                self.logger.info("using DDP static_graph")
            dist_model = DistributedDataParallel(
                model,
                find_unused_parameters=self.config.find_unused_params,
                # device_ids=[self.device.index] if self.device.type == "cuda" else None,  # added for completeness
                static_graph=self.config.static_graph,
            )
        else:
            # DDP broadcasts weights from rank0 to other ranks but raises an error if no param requires grad
            # workaround: temporarily unfreeze one parameter if all parameters are frozen to broadcast weights
            self.logger.info("not wrapping into DDP (no trainable parameters) -> only broadcast parameters")
            first_param = next(model.parameters())
            first_param.requires_grad = True
            dist_model = DistributedDataParallel(model)
            first_param.requires_grad = False

        self.logger.info("replacing BatchNorm layers with SyncBatchNorm")
        return torch.nn.SyncBatchNorm.convert_sync_batchnorm(dist_model)  # type: ignore

    def wrap_compile(self, ddp_model: ModelBase | DistributedDataParallel) -> torch.nn.Module:
        """Wrap the model with torch.compile."""
        if not self.config.use_torch_compile or os.name == "nt":
            return ddp_model
        if is_distributed():
            if self.config.static_graph:
                self.logger.warning("torch.compile static_graph=True is not supported -> disable torch.compile")
                return ddp_model
        self.logger.info("wrapping model with torch.compile")
        compiled = torch.compile(ddp_model)
        # torch.compile should return a torch.nn.Module
        if not isinstance(compiled, torch.nn.Module):
            raise TypeError("torch.compile did not return a torch.nn.Module")
        return compiled

    def train(self, model: ModelBase) -> None:
        """Train the model."""

        self.callbacks = self.get_all_callbacks(model)
        iterator_callbacks = [
            callback for callback in self.callbacks if isinstance(callback, PeriodicDataIteratorCallback)
        ]

        model = self._prepare_model(model)
        dist_model = self.wrap_model(model).to(model.device)

        batch_size, accumulation_steps, train_batches_per_epoch = self._prepare_batch_size()

        data_loader = self.get_data_loader(iterator_callbacks=iterator_callbacks, batch_size=batch_size)
        dist_model.eval()
        self.call_before_training(self.callbacks)
        dist_model.train()

        self._train(
            model=model,
            dist_model=dist_model,
            batch_size=batch_size,
            accumulation_steps=accumulation_steps,
            data_loader=data_loader,
            train_batches_per_epoch=train_batches_per_epoch,
            periodic_callbacks=[
                callback_instance
                for callback_instance in self.callbacks
                if isinstance(callback_instance, PeriodicCallback)
            ],
        )

        dist_model.eval()
        self.call_after_training(callbacks=self.callbacks)
        self.log_writer.finish()

    def _train(
        self,
        model: ModelBase,
        dist_model: torch.nn.Module,
        batch_size: int,
        accumulation_steps: int,
        data_loader: torch.utils.data.DataLoader,
        train_batches_per_epoch: int,
        periodic_callbacks: list[PeriodicCallback],
    ) -> None:
        self.logger.info("Running training loop")

        context_filter = TrainingContextFilter(self.update_counter)
        # Filter on the root logger is not propagated to child loggers, so we add it to the handlers
        for handler in logging.getLogger().handlers:
            handler.addFilter(context_filter)

        try:
            self.logger.debug("initializing dataloader workers")
            data_iter = iter(data_loader)
            self.logger.debug("initialized dataloader workers")
            while True:
                should_stop = self._run_epoch(
                    model=model,
                    dist_model=dist_model,
                    batch_size=batch_size,
                    accumulation_steps=accumulation_steps,
                    data_iter=data_iter,
                    train_batches_per_epoch=train_batches_per_epoch,
                    periodic_callbacks=periodic_callbacks,
                )

                if should_stop:
                    break
        finally:
            for handler in logging.getLogger().handlers:
                handler.removeFilter(context_filter)

    @torch.no_grad()
    def _run_periodic_callbacks(
        self,
        periodic_callbacks: list[PeriodicCallback],
        model: ModelBase,
        dist_model: torch.nn.Module,
        data_iter: Iterator,
        batch_size: int,
        end_of_epoch: bool = False,
    ) -> bool:
        iterator_callback_args = dict(
            trainer_model=dist_model,
            data_iter=map(BaseTrainer.drop_metadata, data_iter),
            batch_size=batch_size,
        )
        early_exit = False
        first_error = None
        for callback in periodic_callbacks:
            try:
                if end_of_epoch:
                    callback.after_epoch(
                        update_counter=self.update_counter,
                        **(iterator_callback_args if isinstance(callback, PeriodicDataIteratorCallback) else {}),
                    )
                else:
                    callback.after_update(
                        update_counter=self.update_counter,
                        **(iterator_callback_args if isinstance(callback, PeriodicDataIteratorCallback) else {}),
                    )
            except EarlyStopIteration:
                self.logger.info(f"Callback {callback} requested early stop of training")
                early_exit = True
            except Exception as e:
                # log first error and continue with other callbacks
                # this way all callbacks get a chance to run their after_update
                # reraise first error after all callbacks have run
                if first_error is None:
                    first_error = e
                self.logger.exception(f"Error in callback {callback}, continuing with other callbacks before exiting")

        if end_of_epoch or not self.update_counter.is_full_epoch:
            self.log_writer.flush()

        if first_error is not None:
            try:
                self.checkpoint_writer.save(
                    model=model, checkpoint_tag=f"{self.update_counter.cur_iteration}.error", trainer=self
                )
            except Exception:
                self.logger.exception("Failed to save error checkpoint")
            raise first_error

        if early_exit:
            self.checkpoint_writer.save(
                model=model, checkpoint_tag=f"{self.update_counter.cur_iteration}.early_exit", trainer=self
            )

        return early_exit

    @staticmethod
    def drop_metadata(data):
        if isinstance(data, dict):
            meta_keys = [k for k in data.keys() if k.startswith("__meta")]
            for k in meta_keys:
                data.pop(k)
        return data

    def _run_epoch(
        self,
        model: ModelBase,
        dist_model: torch.nn.Module,
        batch_size: int,
        accumulation_steps: int,
        data_iter: Iterator,
        train_batches_per_epoch: int,
        periodic_callbacks: list[PeriodicCallback],
    ) -> bool:
        """Run a single epoch. Returns True if training should stop."""
        iter_step = -1
        times: dict[str, float] = defaultdict(float)

        while True:
            # check end of epoch
            remaining_batches = train_batches_per_epoch - (iter_step + 1)
            if remaining_batches < accumulation_steps:
                # InterleavedSampler already have the next batches preloaded -> skip them
                for _ in range(remaining_batches):
                    _ = next(data_iter)
                break

            is_last_update_in_epoch = remaining_batches - accumulation_steps < accumulation_steps

            # Run accumulation steps
            for _ in range(accumulation_steps):
                with Stopwatch() as sw:
                    batch = next(data_iter)
                iter_step += 1
                if iter_step % accumulation_steps == 0:
                    times.clear()
                    model.optimizer_schedule_step()
                times[TRAINING_DATA_WAIT_TIME] += sw.elapsed_seconds
                for key in batch:
                    if key.startswith("__meta_time_"):
                        times[key[len("__meta_time_") :]] += float(batch[key])
                batch = self.drop_metadata(batch)
                batch = move_items_to_device(self.device, batch)

                dist_model.train()
                with Stopwatch() as sw:
                    losses, additional_outputs = self.update(
                        batch=batch,
                        dist_model=dist_model,
                        model=model,
                        accumulation_steps=accumulation_steps,
                    )
                times[TRAINING_UPDATE_TIME] += sw.elapsed_seconds

                with torch.no_grad():
                    for callback in periodic_callbacks:
                        callback.track_after_accumulation_step(
                            update_counter=self.update_counter,
                            batch=batch,
                            losses=losses,
                            update_outputs=additional_outputs,
                            accumulation_steps=accumulation_steps,
                            accumulation_step=iter_step,
                        )
                additional_outputs = None
                batch = None

            # Advance counter
            self.update_counter.add_samples(self.config.effective_batch_size)
            self.update_counter.next_update()
            if is_last_update_in_epoch:
                self.update_counter.next_epoch()

            # Run callbacks after update
            dist_model.eval()
            with torch.no_grad():
                for callback in periodic_callbacks:
                    callback.track_after_update_step(
                        update_counter=self.update_counter,
                        times=times,
                    )

            early_exit = self._run_periodic_callbacks(
                periodic_callbacks=periodic_callbacks,
                model=model,
                dist_model=dist_model,
                data_iter=data_iter,
                batch_size=batch_size,
            )
            if early_exit:
                return True

            # Check end of training
            if self.update_counter.is_finished:
                self._skip_remaining_batches(data_iter, remaining_batches, accumulation_steps, batch_size)

        return self._handle_end_of_epoch(
            model=model,
            dist_model=dist_model,
            batch_size=batch_size,
            periodic_callbacks=periodic_callbacks,
            data_iter=data_iter,
        )

    def _skip_remaining_batches(
        self, data_iter, remaining_batches: int, accumulation_steps: int, batch_size: int
    ) -> None:
        """Skip remaining preloaded batches after training ends."""
        if (
            hasattr(data_iter, "batch_sampler")
            and hasattr(data_iter.batch_sampler, "sampler")
            and hasattr(data_iter.batch_sampler.sampler, "epochs")
            and data_iter.batch_sampler.sampler.epochs is not None
        ):
            for _ in range(remaining_batches - accumulation_steps):
                _ = next(data_iter)

        if (
            hasattr(data_iter, "batch_sampler")
            and hasattr(data_iter.batch_sampler, "sampler")
            and hasattr(data_iter.batch_sampler.sampler, "samples")
            and data_iter.batch_sampler.sampler.samples is not None
        ):
            total_batches = int(data_iter.batch_sampler.sampler.samples / batch_size)
            for _ in range(total_batches % accumulation_steps):
                _ = next(data_iter)

    def _handle_end_of_epoch(
        self,
        model: ModelBase,
        dist_model: torch.nn.Module,
        batch_size: int,
        periodic_callbacks: list[PeriodicCallback],
        data_iter,
    ) -> bool:
        """Handle end of epoch callbacks and checks. Returns True if training should stop."""
        if not self.update_counter.is_full_epoch:
            return False

        early_exit = self._run_periodic_callbacks(
            periodic_callbacks=periodic_callbacks,
            model=model,
            dist_model=dist_model,
            data_iter=data_iter,
            batch_size=batch_size,
            end_of_epoch=True,
        )

        # Check end of training
        return self.update_counter.is_finished or early_exit

    def update(
        self,
        batch: dict[str, Tensor],
        dist_model: torch.nn.Module,
        model: ModelBase | None = None,
        training: bool = True,
        accumulation_steps: int = 1,
        iter_step: int = 0,
        **kwargs,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor] | None]:
        """Perform forward and backward pass."""

        if dist_model.training != training:
            raise ValueError(
                f"model training attribute ({dist_model.training}) does not match training argument ({training})"
            )

        # Forward pass
        with self.autocast_context:
            trainer_result = self.train_step(batch, model=dist_model)
        if not isinstance(trainer_result, TrainerResult):
            raise TypeError("model forward needs to return a TrainerResult")
        if training:
            self._gradient_step(
                total_loss=trainer_result.total_loss,
                model=model if model is not None else dist_model.model,  # type: ignore[arg-type]
                accumulation_steps=accumulation_steps,
                iter_step=iter_step,
                **kwargs,
            )

        all_losses = dict(total=trainer_result.total_loss.detach())
        if trainer_result.losses_to_log is not None:
            all_losses.update({k: v.detach() for k, v in trainer_result.losses_to_log.items()})
        return all_losses, trainer_result.additional_outputs

    def _gradient_step(
        self,
        total_loss: Tensor,
        model: ModelBase,
        accumulation_steps: int,
        iter_step: int,
        retain_graph: bool = False,
    ) -> None:
        if model.is_frozen:
            return

        total_loss = total_loss / accumulation_steps

        if self.config.skip_nan_loss:
            reduced_loss = all_gather_nograd(total_loss)
            if torch.any(torch.isnan(reduced_loss)).item():
                self.logger.info(f"encountered nan loss -> skip (counter: {self.skip_nan_loss_counter})")
                self.skip_nan_loss_counter += 1
                if self.skip_nan_loss_counter > self.config.skip_nan_loss_max_count:
                    raise RuntimeError(f"encountered {self.config.skip_nan_loss_max_count} nan losses in a row")

                self._skip_nan_step = True

            elif self.skip_nan_loss_counter > 0:
                self.logger.info(f"encountered valid loss after {self.skip_nan_loss_counter} nan losses")
                self.skip_nan_loss_counter = 0

        # Backward pass
        if not self._skip_nan_step:
            self.grad_scaler.scale(total_loss).backward(retain_graph=retain_graph)
            self._warn_unused_params(model)

        if (iter_step + 1) % accumulation_steps == 0:
            # only take optimizer step every `accumulation_steps` steps

            if not self._skip_nan_step:
                # skip entire step if all accumulation steps were skipped
                model.optimizer_step(self.grad_scaler)

            model.optimizer_zero_grad()
            # reset skip_nan_step
            self._skip_nan_step = False

    def _warn_unused_params(self, model: ModelBase):
        if self._has_logged_unused_params or not is_rank0():
            return

        unused_param_names = model.nograd_paramnames
        if len(unused_param_names) > 0:
            if is_distributed():
                self.logger.error(
                    f"Found {len(unused_param_names)} unused parameters, this can cause errors with DistributedDataParallel (params: {', '.join(unused_param_names)})"
                )
            else:
                self.logger.warning(f"{len(unused_param_names)} unused parameters: {', '.join(unused_param_names)}")
        self._has_logged_unused_params = True

    def _prepare_batch_size(self) -> tuple[int, int, int]:
        batch_size, accumulation_steps = self._calculate_batch_size_and_accumulation_steps()
        if accumulation_steps > 1 and self.end_checkpoint.update is not None:
            raise NotImplementedError(
                "InterleavedSampler counts every batch as update "
                "-> accumulation steps not supported with update-based end_checkpoint"
            )
        # set accumulation steps in model (e.g. for AsyncBatchNorm it is initialized with accumulation_steps=1
        # but needs to be updated once the actual accumulation_steps are known)
        train_dataset = self.data_container.get_dataset("train")  # mode is not needed because only size is relevant
        train_batches_per_epoch = int(len(train_dataset) / self.config.effective_batch_size * accumulation_steps)
        self.logger.info(
            f"Calculated local batch_size: {batch_size}, accumulation_steps: {accumulation_steps} "
            f"(effective_batch_size={self.config.effective_batch_size}), "
            f"train_batches per epoch: {train_batches_per_epoch} "
            f"(world_size={get_world_size()})"
        )
        return batch_size, accumulation_steps, train_batches_per_epoch

    @torch.no_grad()
    def call_before_training(self, callbacks: list[CallbackBase]) -> None:
        """Hook that is called before training starts."""
        self.logger.info("Running before_training callbacks")
        for callback in callbacks:
            callback.before_training(update_counter=self.update_counter)
            self.logger.debug(f"Executing {callback}")

    @torch.no_grad()
    def call_after_training(self, callbacks: list[CallbackBase]) -> None:
        """Hook that is called after training ends."""
        self.logger.info("Finished training. Running after_training callbacks")
        for callback in callbacks:
            callback.after_training(update_counter=self.update_counter)
            self.logger.debug(f"Executing {callback}")
        self.log_writer.flush()

    def eval(self, model: ModelBase) -> None:
        """Run evaluation by executing all configured callbacks."""
        self.logger.info("Starting evaluation")
        callbacks = self.get_user_callbacks(model, evaluation=True)
        model = self._prepare_model(model)
        dist_model = self.wrap_model(model).to(model.device).eval()
        iterator_callbacks = [callback for callback in callbacks if isinstance(callback, PeriodicDataIteratorCallback)]
        batch_size, _, _ = self._prepare_batch_size()

        data_loader = self.get_data_loader(
            iterator_callbacks=iterator_callbacks, batch_size=batch_size, evaluation=True
        )
        data_iter = iter(data_loader)

        for callback in callbacks:
            if not isinstance(callback, PeriodicCallback):
                continue
            self.logger.info(f"Running periodic callback: {callback}")
            iterator_callback_args = (
                dict(
                    trainer_model=dist_model,
                    data_iter=map(BaseTrainer.drop_metadata, data_iter),
                    batch_size=batch_size,
                )
                if isinstance(callback, PeriodicDataIteratorCallback)
                else {}
            )
            callback.at_eval(self.update_counter, **iterator_callback_args)

    @property
    def total_training_updates(self) -> int:
        if self.end_checkpoint.epoch is not None:
            return self.end_checkpoint.epoch * self.update_counter.updates_per_epoch
        elif self.end_checkpoint.update is not None:
            return self.end_checkpoint.update
        elif self.end_checkpoint.sample is not None:
            return self.end_checkpoint.sample // self.update_counter.effective_batch_size
        else:
            raise ValueError(
                "end_checkpoint needs to have either epoch, update or sample defined to calculate total training updates"
            )
