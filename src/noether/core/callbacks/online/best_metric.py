#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

from noether.core.callbacks.periodic import PeriodicCallback
from noether.core.schemas.callbacks import BestMetricCallbackConfig


class BestMetricCallback(PeriodicCallback):
    """A callback that keeps track of the best metric value over a training run for a certain metric (i.e., source_metric_key) while also logging one or more target metrics.

    For example, track the test loss the epoch with the best validation loss to simulate early stopping.

    Example config:

    .. code-block:: yaml

        - kind: noether.core.callbacks.BestMetricCallback
          every_n_epochs: 1
          source_metric_key: loss/val/total
          target_metric_keys:
            -  loss/test/total

    In this example, whenever a new best validation loss is found, the corresponding test loss is logged under the key ``loss/test/total/at_best/loss/val/total``.
    """

    def __init__(
        self,
        callback_config: BestMetricCallbackConfig,
        **kwargs,
    ):
        """

        Args:
            callback_config: Configuration for the callback. See
                :class:`~noether.core.schemas.callbacks.BestMetricCallbackConfig`
                for available options including source and target metric keys.
            **kwargs: Additional keyword arguments provided to the parent class.
        """
        super().__init__(callback_config=callback_config, **kwargs)

        if callback_config.target_metric_keys is None and callback_config.optional_target_metric_keys is None:
            raise ValueError("At least one of 'target_metric_keys' or 'optional_target_metric_keys' must be provided.")

        self.source_metric_key = callback_config.source_metric_key
        self.target_metric_keys = callback_config.target_metric_keys
        self.optional_target_metric_keys = callback_config.optional_target_metric_keys
        self.higher_is_better = self.metric_property_provider.higher_is_better(self.source_metric_key)
        self.best_metric_value = -float("inf") if self.higher_is_better else float("inf")
        self.previous_log_values: dict[str, Any] = dict()

    def _is_new_best(self, metric_value: float) -> bool:
        return (
            (metric_value > self.best_metric_value)
            if self.higher_is_better
            else (metric_value < self.best_metric_value)
        )

    def _validate_key_exists(self, key: str, key_type: str):
        """Helper to validate key existence in the log_cache."""
        if self.writer.log_cache is None or key not in self.writer.log_cache:
            raise ValueError(
                f"Couldn't find {key_type} '{key}' (valid metric keys="
                f"{list(self.writer.log_cache.keys() if self.writer.log_cache is not None else [])}) -> make sure the callback that produces the metric_key is "
                f"called at the same (or higher) frequency and is ordered before the {type(self).__name__}"
            )

    def _log_and_cache_metric(self, metric_key: str, value: float) -> None:
        """Helper to log the metric and update the cache."""
        log_key = f"{metric_key}/at_best/{self.source_metric_key}"

        self.writer.add_scalar(
            key=log_key,
            value=value,
            logger=self.logger,
            format_str=".6f",
        )
        self.previous_log_values[log_key] = value

    # noinspection PyMethodOverriding
    def periodic_callback(self, **__) -> None:
        # check that all mandatory keys are in the log_cache
        self._validate_key_exists(self.source_metric_key, "source_metric_key")
        assert self.writer.log_cache is not None

        for target_key in self.target_metric_keys or []:
            self._validate_key_exists(target_key, "target_metric_key")

        source_metric_value = self.writer.log_cache[self.source_metric_key]

        # Check for improvement:
        if self._is_new_best(source_metric_value):
            # log source_metric_key improvement
            self.logger.info(
                f"New best model ({self.source_metric_key}): {self.best_metric_value} --> {source_metric_value}",
            )

            # Collect all keys to log (Mandatory + Existing Optionals):
            keys_to_log = (self.target_metric_keys or []).copy()

            # Add optional keys only if they currently exist in the cache:
            if self.optional_target_metric_keys:
                keys_to_log.extend(key for key in self.optional_target_metric_keys if key in self.writer.log_cache)

            for target_key in keys_to_log:
                self._log_and_cache_metric(target_key, self.writer.log_cache[target_key])

            self.best_metric_value = source_metric_value
        else:
            # Fallback: Log previous values to keep graphs smooth:
            for key, value in self.previous_log_values.items():
                self.writer.add_scalar(
                    key=key,
                    value=value,
                    logger=self.logger,
                    format_str=".6f",
                )
