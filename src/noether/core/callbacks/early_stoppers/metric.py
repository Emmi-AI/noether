#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.callbacks.early_stoppers.base import EarlyStopperBase
from noether.core.schemas.callbacks import MetricEarlyStopperConfig


class MetricEarlyStopper(EarlyStopperBase):
    """Early stopper (training) based on a metric value to be monitored.

    Example config:

    .. code-block:: yaml

        - kind: noether.core.callbacks.MetricEarlyStopper
          every_n_epochs: 1
          metric_key: loss/val/total
          tolerance: 0.10
          name: MetricEarlyStopper
   """

    def __init__(
        self,
        callback_config: MetricEarlyStopperConfig,
        **kwargs,
    ):
        """

        Args:
            callback_config: Configuration for the callback. See
                :class:`~noether.core.schemas.callbacks.MetricEarlyStopperConfig`
                for available options including metric key and tolerance.
            **kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__(callback_config, **kwargs)
        self.metric_key = callback_config.metric_key
        self.higher_is_better = self.metric_property_provider.higher_is_better(callback_config.metric_key)
        self.tolerance = callback_config.tolerance
        self.tolerance_counter = 0
        self.best_metric = -float("inf") if self.higher_is_better else float("inf")

    def _metric_improved(self, cur_metric):
        if self.higher_is_better:
            return cur_metric > self.best_metric
        return cur_metric < self.best_metric

    def _should_stop(self, **_):
        if self.writer.log_cache is None or self.metric_key not in self.writer.log_cache:
            valid_metric_keys = list(self.writer.log_cache.keys()) if self.writer.log_cache is not None else []
            raise ValueError(
                f"couldn't find metric_key {self.metric_key} (valid metric_keys={valid_metric_keys}) -> "
                "make sure every_n_epochs/every_n_updates/every_n_samples is aligned with the corresponding callback"
            )

        cur_metric = self.writer.log_cache[self.metric_key]

        if self._metric_improved(cur_metric):
            self.logger.info(f"{self.metric_key} improved: {self.best_metric} --> {cur_metric}")
            self.best_metric = cur_metric
            self.tolerance_counter = 0
        else:
            self.tolerance_counter += 1
            cmp_str = "<=" if self.higher_is_better else ">="
            stop_training_str = " --> stop training" if self.tolerance_counter >= self.tolerance else ""
            self.logger.info(
                f"{self.metric_key} stagnated: {self.best_metric} {cmp_str} {cur_metric} "
                f"({self.tolerance_counter}/{self.tolerance}){stop_training_str}"
            )

        return self.tolerance_counter >= self.tolerance
