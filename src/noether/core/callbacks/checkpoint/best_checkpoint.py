#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Annotated, Any, Literal

from pydantic import Field
from torch.utils.data import DataLoader, DistributedSampler

from noether.core.callbacks.base import CallbackBase, CallBackBaseConfig
from noether.core.callbacks.periodic import IntervalType, PeriodicCallback, PeriodicDataIteratorCallback
from noether.core.factory import Factory
from noether.core.schemas.lib import Discriminated
from noether.core.writers import PrefixedLogWriter


class BestCheckpointCallbackConfig(CallBackBaseConfig):
    name: Literal["BestCheckpointCallback"] = Field("BestCheckpointCallback", frozen=True)

    metric_key: str = Field(...)
    """"The key of the metric to be used for checking the best model."""
    save_frozen_weights: bool = Field(True)
    """Whether to also save the frozen weights of the model."""
    tolerances: list[int] | None = Field(
        None,
    )
    """"If provided, this callback will produce multiple best models which differ in the amount of intervals they allow the metric to not improve. For example, tolerance=[5] with every_n_epochs=1 will store a checkpoint where at most 5 epochs have passed until the metric improved. Additionally, the best checkpoint over the whole training will always be stored (i.e., tolerance=infinite). When setting different tolerances, one can evaluate different early stopping configurations with one training run."""
    model_names: list[str] | None = Field(None)
    """Which model name to save (e.g., if only the encoder of an autoencoder should be stored, one could pass model_name='encoder' here). This only applies when training a CompositeModel. If None, all models are saved."""
    eval_callbacks: list[Annotated[Any, Discriminated(CallBackBaseConfig)]] | None = Field(None)
    """Optional nested callbacks to dispatch whenever a new best model is detected. Each child's metric keys
    are automatically prefixed with ``best=<metric_key>/`` (slashes in the metric key are replaced with dots)
    so they don't collide with the live-model metrics. Children are invoked via their ``at_eval`` hook, which
    bypasses their own schedule — the trigger is the new-best event, not the child's ``every_n_*``. Tolerance-
    exceeded saves do not trigger children. ``before_training`` and ``after_training`` are forwarded
    unconditionally so children can initialize and finalize cleanly.

    ``PeriodicDataIteratorCallback`` children get a dedicated ``DataLoader`` built from their
    ``sampler_config``; they are *not* registered on the shared
    :class:`~noether.data.samplers.InterleavedSampler`. This means a child's ``every_n_*`` is irrelevant
    here (only the ``dataset_key`` / ``batch_size`` / ``pipeline`` matter) and the child's schedule does
    not need to match this callback's."""


class BestCheckpointCallback(PeriodicCallback):
    """Callback to save the best model based on a metric.

    This callback monitors a specified metric and saves the model checkpoint whenever
    a new best value is achieved. It supports storing different model components when using a composite model and can save checkpoints at different tolerance thresholds.

    Example config:

    .. code-block:: yaml

        callbacks:
          - kind: noether.core.callbacks.BestCheckpointCallback
            name: BestCheckpointCallback
            every_n_epochs: 1
            metric_key: loss/val/total
            model_names:  # only applies when training a CompositeModel
              - encoder
            eval_callbacks:
              - kind: noether.training.callbacks.OfflineLossCallback
                every_n_epochs: 1  # ignored; the parent triggers on new-best
                dataset_key: test
    """

    def __init__(
        self,
        callback_config: BestCheckpointCallbackConfig,
        **kwargs,
    ):
        """

        Args:
            callback_config: Configuration for the callback. See
                :class:`~noether.core.schemas.callbacks.BestCheckpointCallbackConfig`
                for available options including metric key, model names, and tolerance settings.
            **kwargs: Additional arguments passed to the parent class.
        """

        super().__init__(callback_config=callback_config, **kwargs)
        self.metric_key = callback_config.metric_key
        self.model_names = callback_config.model_names or []
        self.higher_is_better = self.metric_property_provider.higher_is_better(self.metric_key)
        self.best_metric_value = -float("inf") if self.higher_is_better else float("inf")
        self.save_frozen_weights = callback_config.save_frozen_weights

        # save multiple best models based on tolerance
        self.tolerances_is_exceeded = dict.fromkeys(callback_config.tolerances or [], False)
        self.tolerance_counter = 0
        self.metric_at_exceeded_tolerance: dict[float, float] = {}

        # Build nested eval callbacks. Each child gets a PrefixedLogWriter so its metric keys are namespaced
        # as ``best=<metric_key>/<original_key>``. Children are dispatched on new-best detection (bypassing
        # their own schedule via ``at_eval``).
        self.eval_callbacks: list[PeriodicCallback] = []
        eval_callback_configs = getattr(callback_config, "eval_callbacks", None)
        if eval_callback_configs:
            prefix = f"best={self.metric_key.replace('/', '.')}"
            child_kwargs = {
                **kwargs,
                "log_writer": PrefixedLogWriter(inner=kwargs["log_writer"], prefix=prefix),
            }
            self.eval_callbacks = Factory().create_list(eval_callback_configs, **child_kwargs)

        # Cache one DataLoader per iterator child. Built lazily on first new-best so we don't pay the cost
        # if the metric never improves (e.g. divergence). Reused thereafter — ``iter(loader)`` returns a
        # fresh iterator each time and rebuilds the (deterministic) sampler order.
        self._iterator_child_loaders: dict[int, DataLoader] = {}

    def get_children(self) -> list[CallbackBase]:
        """Non-iterator children only — iterator children are owned end-to-end here and must not be
        registered on the shared :class:`~noether.data.samplers.InterleavedSampler` (we build their
        loaders on dispatch instead). The trainer always passes ``batch_size`` to every
        :class:`~noether.core.callbacks.periodic.PeriodicCallback` hook, so we can build child loaders
        without needing the trainer's iterator-args bundle.
        """
        children: list[CallbackBase] = [
            child for child in self.eval_callbacks if not isinstance(child, PeriodicDataIteratorCallback)
        ]
        return children

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the callback for checkpointing.

        Returns:
            Dictionary containing the best metric value, tolerance tracking state,
            and counter information.
        """
        return dict(
            best_metric_value=self.best_metric_value,
            tolerances_is_exceeded=self.tolerances_is_exceeded,
            tolerance_counter=self.tolerance_counter,
            metric_at_exceeded_tolerance=self.metric_at_exceeded_tolerance,
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the callback state from a checkpoint.

        Note:
            This modifies the input state_dict in place.

        Args:
            state_dict: Dictionary containing the saved callback state.
        """
        self.best_metric_value = state_dict["best_metric_value"]
        self.tolerances_is_exceeded = state_dict["tolerances_is_exceeded"]
        self.tolerance_counter = state_dict["tolerance_counter"]
        self.metric_at_exceeded_tolerance = state_dict["metric_at_exceeded_tolerance"]

    def before_training(self, *, update_counter, **kwargs) -> None:
        """Validate callback configuration before training starts.

        Args:
            update_counter: The training update counter.
            **kwargs: Additional keyword arguments forwarded to child eval callbacks.

        Raises:
            NotImplementedError: If resuming training with tolerances is attempted.
        """
        if len(self.tolerances_is_exceeded) > 0 and update_counter.cur_iteration.sample > 0:
            raise NotImplementedError(f"{type(self).__name__} with tolerances resuming not implemented")

        for child in self.eval_callbacks:
            child.before_training(update_counter=update_counter, **kwargs)

    def _is_new_best_model(self, metric_value):
        """Check if the current metric value is better than the best recorded value.

        Args:
            metric_value: The current metric value to compare.

        Returns:
            True if the current value is better than the best value, False otherwise.
        """
        if self.higher_is_better:
            return metric_value > self.best_metric_value
        return metric_value < self.best_metric_value

    def _build_loader_for_child(self, child: PeriodicDataIteratorCallback, batch_size: int | None) -> DataLoader:
        """Build (and cache) a standalone ``DataLoader`` for an iterator child.

        Uses the child's existing ``sampler_config`` — the sampler, dataset, collate pipeline and batch
        size have already been resolved against the data container during the child's ``__init__``. We
        just hand them to a plain ``DataLoader`` so the child can iterate its dataset without going
        through the shared :class:`~noether.data.samplers.InterleavedSampler`.
        """
        cache_key = id(child)
        if cache_key in self._iterator_child_loaders:
            return self._iterator_child_loaders[cache_key]

        config = child.sampler_config
        sampler = config.sampler
        # ``sampler`` is typed as ``SizedIterable`` in ``SamplerIntervalConfig``; in practice it's either
        # a torch ``DistributedSampler`` (has ``dataset``) or a ``SequentialSampler`` (has ``data_source``)
        # — both built by ``PeriodicDataIteratorCallback._create_sampler_config``.
        dataset = sampler.dataset if isinstance(sampler, DistributedSampler) else sampler.data_source  # type: ignore[attr-defined]
        loader = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size or batch_size,
            sampler=sampler,
            collate_fn=config.pipeline,
            persistent_workers=True,
        )
        self._iterator_child_loaders[cache_key] = loader
        return loader

    def _dispatch_children(self, *, interval_type: IntervalType, update_counter=None, **kwargs) -> None:
        """Dispatch each child against the live (= new best) model.

        Iterator children receive a fresh ``data_iter`` built from their own dataset (the trainer's
        shared ``data_iter`` is *not* used). ``trainer_model`` / ``batch_size`` arrive via ``kwargs``
        from the trainer — both are always passed to every :class:`PeriodicCallback` hook.

        Children are invoked via :meth:`periodic_callback` (not :meth:`at_eval`) so they observe the
        parent's actual ``interval_type`` (e.g. ``"epoch"`` / ``"update"``) instead of a synthetic
        ``"eval"`` — matching the context in which the new-best event was raised.
        """
        batch_size = kwargs.get("batch_size")
        for child in self.eval_callbacks:
            child_kwargs = dict(kwargs)
            if isinstance(child, PeriodicDataIteratorCallback):
                loader = self._build_loader_for_child(child, batch_size=batch_size)
                child_kwargs["data_iter"] = iter(loader)
            child.periodic_callback(
                interval_type=interval_type,
                update_counter=update_counter,
                **child_kwargs,
            )

    # noinspection PyMethodOverriding
    def periodic_callback(self, *, interval_type: IntervalType, **kwargs) -> None:
        """Execute the periodic callback to check and save best model.

        This method is called at the configured frequency (e.g., every N epochs).
        It checks if the current metric value is better than the previous best,
        and if so, saves the model checkpoint. Also tracks tolerance-based checkpoints.

        When a new best is detected, child eval callbacks (if configured) are dispatched against the live
        (newly-best) model. Iterator children iterate their own :class:`~torch.utils.data.DataLoader`
        (built on first use) — they do **not** consume from the trainer's shared ``data_iter``.

        On ``interval_type="eval"`` (post-training eval, where the trainer loads the saved best checkpoint
        into the live model and calls every callback's ``at_eval``), children are dispatched
        unconditionally so they evaluate the loaded best model. No checkpoint save / tolerance bookkeeping
        runs in eval mode (the in-memory ``best_metric_value`` starts at ±inf in a fresh eval process).

        Raises:
            KeyError: If the log cache is empty or the metric key is not found.
        """
        if interval_type == "eval":
            # Post-training eval: model is already the loaded best; just dispatch children.
            self._dispatch_children(interval_type=interval_type, **kwargs)
            return
        if self.writer.log_cache is None:
            raise KeyError("Log cache is empty, can't retrieve metric value.")
        if self.metric_key not in self.writer.log_cache:
            raise KeyError(
                f"couldn't find metric_key {self.metric_key} (valid metric keys={list(self.writer.log_cache.keys())}) -> "
                "make sure the callback that produces the metric_key is called at the same (or higher) frequency and "
                f"is ordered before the {type(self).__name__}"
            )
        metric_value = self.writer.log_cache[self.metric_key]

        if self._is_new_best_model(metric_value):
            # one could also track the model and save it after training
            # this is better in case runs crash or are terminated
            # the runtime overhead is negligible
            self.logger.info(f"new best model ({self.metric_key}): {self.best_metric_value} --> {metric_value}")
            self.checkpoint_writer.save(
                model=self.model,
                checkpoint_tag=f"best_model.{self.metric_key.replace('/', '.')}",
                save_optim=False,
                model_names_to_save=self.model_names,
            )
            self.best_metric_value = metric_value

            # Reset the tolerance flag and the exceeded flags so tolerance tracking starts over:
            self.tolerance_counter = 0
            self.tolerances_is_exceeded = dict.fromkeys(self.tolerances_is_exceeded, False)

            self._dispatch_children(interval_type=interval_type, **kwargs)
        else:
            self.tolerance_counter += 1
            for tolerance, is_exceeded in self.tolerances_is_exceeded.items():
                if is_exceeded or self.tolerance_counter <= tolerance:
                    continue
                # Check if counter is STRICTLY greater than tolerance:
                self.tolerances_is_exceeded[tolerance] = True
                self.metric_at_exceeded_tolerance[tolerance] = metric_value

                # Only save checkpoint if we got a better value at least once.
                if self.best_metric_value in [float("inf"), -float("inf")]:
                    continue

                self.checkpoint_writer.save(
                    model=self.model,
                    checkpoint_tag=f"best_model.{self.metric_key.replace('/', '.')}.tolerance={tolerance}",
                    save_optim=False,
                    model_names_to_save=self.model_names,
                )

    def after_training(self, **kwargs) -> None:
        """Log the best metric values at different tolerance thresholds after training completes.

        Args:
            **kwargs: Additional keyword arguments forwarded to child eval callbacks.
        """
        # best metric doesn't need to be logged as it is summarized anyways
        for tolerance, value in self.metric_at_exceeded_tolerance.items():
            self.logger.info(f"best {self.metric_key} with tolerance={tolerance}: {value}")

        for child in self.eval_callbacks:
            child.after_training(**kwargs)
