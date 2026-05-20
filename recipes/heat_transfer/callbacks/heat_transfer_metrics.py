#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import torch
from pydantic import Field

from noether.core.callbacks.periodic import PeriodicDataIteratorCallback, PeriodicDataIteratorCallbackConfig

EVALUATION_FIELDS = [
    "volume_velocity",
    "volume_temperature",
    "volume_pressure",
]

METRIC_SUFFIX_TARGET = "_target"
METRIC_PREFIX = "denormalized/"


class HeatTransferMetricsCallbackConfig(PeriodicDataIteratorCallbackConfig):
    """Configuration for heat transfer evaluation metrics callback."""

    kind: str | None = "heat_transfer.callbacks.HeatTransferMetricsCallback"

    forward_properties: list[str] = []
    """List of properties in the dataset to be forwarded during inference."""
    batch_size: int = Field(1)
    """Batch size for evaluation. Currently only batch_size=1 is supported."""


class HeatTransferMetricsCallback(PeriodicDataIteratorCallback):
    """Callback for computing denormalized RMSE and nRMSE metrics on heat transfer predictions.

    Periodically evaluates model performance by computing RMSE and normalized RMSE
    (nRMSE = RMSE / RMS(target)) for velocity, temperature, and pressure fields.
    All metrics are computed in denormalized (physical) space.
    """

    def __init__(self, callback_config: HeatTransferMetricsCallbackConfig, **kwargs):
        super().__init__(callback_config, **kwargs)

        self.dataset_normalizers = self.data_container.get_dataset(self.dataset_key).normalizers
        self.forward_properties = callback_config.forward_properties

    def _compute_metrics(
        self, denormalized_predictions: torch.Tensor, denormalized_targets: torch.Tensor, field_name: str
    ) -> dict[str, torch.Tensor]:
        """Compute RMSE and nRMSE for a given field.

        nRMSE is defined as RMSE / RMS(target), giving a dimensionless relative error.
        """
        delta = denormalized_predictions - denormalized_targets

        rmse = (delta**2).mean().sqrt()
        metrics = {f"{field_name}_rmse": rmse}

        # nRMSE = RMSE / RMS(target) — skip if target RMS is near zero
        target_rms = (denormalized_targets**2).mean().sqrt()
        if target_rms > 1e-8:
            metrics[f"{field_name}_nrmse"] = rmse / target_rms
        else:
            self.logger.warning(f"Target RMS too small for {field_name}, skipping nRMSE")

        return metrics

    def process_data(self, batch: dict[str, torch.Tensor], **_) -> dict[str, torch.Tensor]:
        """Run model inference and compute denormalized metrics for all fields."""
        forward_inputs = {k: v for k, v in batch.items() if k in self.forward_properties}
        with self.trainer.autocast_context:
            model_outputs = self.model(**forward_inputs)

        metrics = {}
        for field in EVALUATION_FIELDS:
            target = batch.get(f"{field}{METRIC_SUFFIX_TARGET}")
            prediction = model_outputs.get(field)

            if prediction is None or target is None:
                continue

            data_container = self.data_container.get_dataset(self.dataset_key)
            denorm_pred = data_container.denormalize(field, prediction)
            denorm_target = data_container.denormalize(field, target)
            metrics.update(self._compute_metrics(denorm_pred, denorm_target, field))

        return metrics

    def process_results(self, results: dict[str, torch.Tensor], **_) -> None:
        """Log computed metrics."""
        if not results:
            self.logger.warning(f"No metrics computed for dataset '{self.dataset_key}'")
            return

        for name, metric in results.items():
            metric_key = f"{METRIC_PREFIX}{self.dataset_key}/{name}"
            self.writer.add_scalar(
                key=metric_key,
                value=metric.mean(),
                logger=self.logger,
                format_str=".6f",
            )

        self.logger.debug(f"Logged {len(results)} metrics for dataset '{self.dataset_key}'")
