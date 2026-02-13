#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

from noether.core.callbacks.periodic import PeriodicCallback
from noether.core.schemas.callbacks import BestCheckpointCallbackConfig


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

    def before_training(self, *, update_counter) -> None:
        """Validate callback configuration before training starts.

        Args:
            update_counter: The training update counter.

        Raises:
            NotImplementedError: If resuming training with tolerances is attempted.
        """
        if len(self.tolerances_is_exceeded) > 0 and update_counter.cur_iteration.sample > 0:
            raise NotImplementedError(f"{type(self).__name__} with tolerances resuming not implemented")

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

    # noinspection PyMethodOverriding
    def periodic_callback(self, **_) -> None:
        """Execute the periodic callback to check and save best model.

        This method is called at the configured frequency (e.g., every N epochs).
        It checks if the current metric value is better than the previous best,
        and if so, saves the model checkpoint. Also tracks tolerance-based checkpoints.

        Raises:
            KeyError: If the log cache is empty or the metric key is not found.
        """
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
                checkpoint=f"best_model.{self.metric_key.replace('/', '.')}",
                save_optim=False,
                model_names_to_save=self.model_names,
            )
            self.best_metric_value = metric_value

            # Reset the tolerance flag and the exceeded flags so tolerance tracking starts over:
            self.tolerance_counter = 0
            self.tolerances_is_exceeded = dict.fromkeys(self.tolerances_is_exceeded, False)

            # log tolerance checkpoints
            for tolerance, is_exceeded in self.tolerances_is_exceeded.items():
                if is_exceeded:
                    continue
                self.checkpoint_writer.save(
                    model=self.model,
                    checkpoint=f"best_model.{self.metric_key.replace('/', '.')}.tolerance{tolerance}",
                    save_optim=False,
                    model_names_to_save=self.model_names,
                )
        else:
            self.tolerance_counter += 1
            for tolerance, is_exceeded in self.tolerances_is_exceeded.items():
                if is_exceeded:
                    continue
                # Check if counter is STRICTLY greater than tolerance:
                if self.tolerance_counter > tolerance:
                    self.tolerances_is_exceeded[tolerance] = True
                    self.metric_at_exceeded_tolerance[tolerance] = metric_value

    def after_training(self, **kwargs) -> None:
        """Log the best metric values at different tolerance thresholds after training completes.

        Args:
            **kwargs: Additional keyword arguments (unused).
        """
        # best metric doesn't need to be logged as it is summarized anyways
        for tolerance, value in self.metric_at_exceeded_tolerance.items():
            self.logger.info(f"best {self.metric_key} with tolerance={tolerance}: {value}")
