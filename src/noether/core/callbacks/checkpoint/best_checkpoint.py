#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

from noether.core.callbacks.periodic import PeriodicCallback
from noether.core.schemas.callbacks import BestCheckpointCallbackConfig


class BestCheckpointCallback(PeriodicCallback):
    """Callback to save the best model based on a metric.

    Example config:
        .. code-block:: yaml
            - kind: noether.core.callbacks.BestCheckpointCallback
              name: BestCheckpointCallback
              every_n_epochs: 1
              metric_key: loss/val/total
              model_names: # only applies when training a CompositeModel
                - encoder
    """

    def __init__(
        self,
        callback_config: BestCheckpointCallbackConfig,
        **kwargs,
    ):
        """
        Initializes the BestCheckpointCallback.

        Args:
            callback_config: The configuration for the callback.
            **kwargs: additional arguments passed to the parent class.
        """

        super().__init__(callback_config=callback_config, **kwargs)
        self.metric_key = callback_config.metric_key
        self.model_names = callback_config.model_names or []
        if callback_config.model_name is not None:
            self.model_names.append(callback_config.model_name)
        self.higher_is_better = self.metric_property_provider.higher_is_better(self.metric_key)
        self.best_metric_value = -float("inf") if self.higher_is_better else float("inf")
        self.save_frozen_weights = callback_config.save_frozen_weights

        # save multiple best models based on tolerance
        self.tolerances_is_exceeded = dict.fromkeys(callback_config.tolerances or [], False)
        self.tolerance_counter = 0
        self.metric_at_exceeded_tolerance: dict[float, float] = {}

    def state_dict(self) -> dict[str, Any]:
        """Returns the state of the callback."""
        return dict(
            best_metric_value=self.best_metric_value,
            tolerances_is_exceeded=self.tolerances_is_exceeded,
            tolerance_counter=self.tolerance_counter,
            metric_at_exceeded_tolerance=self.metric_at_exceeded_tolerance,
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Loads the state of the callback into the state_dict. Note that this modifies the input state_dict."""
        self.best_metric_value = state_dict["best_metric_value"]
        self.tolerances_is_exceeded = state_dict["tolerances_is_exceeded"]
        self.tolerance_counter = state_dict["tolerance_counter"]
        self.metric_at_exceeded_tolerance = state_dict["metric_at_exceeded_tolerance"]

    def before_training(self, *, update_counter) -> None:
        if len(self.tolerances_is_exceeded) > 0 and update_counter.cur_iteration.sample > 0:
            raise NotImplementedError(f"{type(self).__name__} with tolerances resuming not implemented")

    def _is_new_best_model(self, metric_value):
        if self.higher_is_better:
            return metric_value > self.best_metric_value
        return metric_value < self.best_metric_value

    # noinspection PyMethodOverriding
    def periodic_callback(self, **_) -> None:
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
            self.tolerance_counter = 0
            # log tolerance checkpoints
            for tolerance, is_exceeded in self.tolerances_is_exceeded.items():
                if is_exceeded:
                    continue
                self.checkpoint_writer.save(
                    model=self.model,
                    checkpoint_tag=f"best_model.{self.metric_key.replace('/', '.')}.tolerance{tolerance}",
                    save_optim=False,
                    model_names_to_save=self.model_names,
                )
        else:
            self.tolerance_counter += 1
            for tolerance, is_exceeded in self.tolerances_is_exceeded.items():
                if is_exceeded:
                    continue
                if tolerance >= self.tolerance_counter:
                    self.tolerances_is_exceeded[tolerance] = True
                    self.metric_at_exceeded_tolerance[tolerance] = metric_value

    def after_training(self, **kwargs) -> None:
        # best metric doesn't need to be logged as it is summarized anyways
        for tolerance, value in self.metric_at_exceeded_tolerance.items():
            self.logger.info(f"best {self.metric_key} with tolerance={tolerance}: {value}")
