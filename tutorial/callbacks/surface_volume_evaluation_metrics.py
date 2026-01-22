#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections import defaultdict

import torch

from noether.core.callbacks.periodic import PeriodicIteratorCallback
from tutorial.schemas.callbacks import SurfaceVolumeEvaluationMetricsCallbackConfig

# Constants
DEFAULT_EVALUATION_MODES = [
    "surface_pressure",
    "surface_friction",
    "volume_velocity",
    "volume_pressure",
    "volume_vorticity",
]

METRIC_SUFFIX_TARGET = "_target"
METRIC_PREFIX_LOSS = "loss/"


class MetricType:
    """Metric type identifiers."""

    MSE = "mse"
    MAE = "mae"
    L2ERR = "l2err"


class SurfaceVolumeEvaluationMetricsCallback(PeriodicIteratorCallback):
    """
    Callback for computing evaluation metrics on surface and volume predictions.

    This callback periodically evaluates model performance by computing MSE, MAE,
    and L2 error metrics for various physical fields (pressure, velocity, friction, etc.).
    Supports both standard and chunked inference for memory efficiency.

    Args:
        callback_config: Configuration for the callback including dataset key,
                        forward properties, and chunking settings
        **kwargs: Additional arguments passed to parent class

    Attributes:
        dataset_key: Identifier for the dataset to evaluate
        evaluation_modes: List of field names to evaluate
        dataset_normalizers: Normalizers for denormalizing predictions
        forward_properties: Properties to pass to model forward
        chunked_inference: Whether to use chunked inference
        chunk_properties: Properties to chunk
        chunk_size: Size of each chunk
        chunk_property: Property to determine chunk count
    """

    def __init__(self, callback_config: SurfaceVolumeEvaluationMetricsCallbackConfig, **kwargs):
        super().__init__(callback_config, **kwargs)

        self._config = callback_config
        self.dataset_key = callback_config.dataset_key
        self.evaluation_modes = DEFAULT_EVALUATION_MODES
        self.dataset_normalizers = self.data_container.get_dataset(self.dataset_key).normalizers
        self.forward_properties = callback_config.forward_properties
        self.chunked_inference = callback_config.chunked_inference
        self.chunk_properties = callback_config.chunk_properties
        self.chunk_size = callback_config.chunk_size
        self.sample_size_property = callback_config.sample_size_property

    def _denormalize(
        self, predictions: torch.Tensor, targets: torch.Tensor, key: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Denormalize predictions and targets using the appropriate normalizer.

        This method finds the specific normalizer for the given key and uses it to denormalize,
        instead of calling pipeline.denormalize which would process the entire pipeline.

        Args:
            predictions: Tensor containing the predictions to denormalize
            targets: Tensor containing the targets to denormalize
            key: Key to identify the normalizer for denormalization

        Returns:
            Tuple of (denormalized_predictions, denormalized_targets)

        Raises:
            KeyError: If no normalizer is found for the given key
        """
        try:
            normalizer = self.dataset_normalizers[key]
        except KeyError as e:
            raise KeyError(
                f"No normalizer found for key '{key}'. Available normalizers: {list(self.dataset_normalizers.keys())}"
            ) from e

        denormalized_predictions = normalizer.inverse(predictions.cpu())
        denormalized_targets = normalizer.inverse(targets.cpu())
        return denormalized_predictions, denormalized_targets

    def _compute_metrics(
        self, denormalized_predictions: torch.Tensor, denormalized_targets: torch.Tensor, field_name: str
    ) -> dict[str, torch.Tensor]:
        """
        Compute evaluation metrics for predictions vs targets.

        Calculates Mean Squared Error (MSE), Mean Absolute Error (MAE),
        and relative L2 error for the given field.

        Args:
            denormalized_predictions: Denormalized prediction tensor
            denormalized_targets: Denormalized target tensor
            field_name: Name of the field being evaluated (used for metric naming)

        Returns:
            Dictionary mapping metric names to computed values
        """
        delta = denormalized_predictions - denormalized_targets

        metrics = {
            f"{field_name}_{MetricType.MSE}": (delta**2).mean(),
            f"{field_name}_{MetricType.MAE}": delta.abs().mean(),
        }

        # L2 relative error (avoid division by zero)
        target_norm = denormalized_targets.norm()
        if target_norm > 1e-8:
            metrics[f"{field_name}_{MetricType.L2ERR}"] = delta.norm() / target_norm
        else:
            self.logger.warning(f"Target norm too small for {field_name}, skipping L2 error")

        return metrics

    def _create_chunked_batch(
        self, batch: dict[str, torch.Tensor], start_idx: int, end_idx: int
    ) -> dict[str, torch.Tensor]:
        """
        Create a batch slice for chunked processing.

        Args:
            batch: Full batch dictionary
            start_idx: Start index for the chunk
            end_idx: End index for the chunk

        Returns:
            Dictionary with chunked tensors for specified properties
        """
        chunked_batch = {}
        for key, value in batch.items():
            if key in self.chunk_properties:
                chunked_batch[key] = value[:, start_idx:end_idx]
            else:
                chunked_batch[key] = value
        return chunked_batch

    def _get_chunk_indices(self, batch_size: int) -> list[tuple[int, int]]:
        """
        Calculate start and end indices for all chunks.

        Args:
            batch_size: Total size of the batch to chunk

        Returns:
            List of (start_idx, end_idx) tuples for each chunk
        """
        indices = []
        n_chunks = batch_size // self.chunk_size

        for chunk_idx in range(n_chunks):
            start = chunk_idx * self.chunk_size
            end = start + self.chunk_size
            indices.append((start, end))

        return indices

    def _chunked_model_inference(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Run model inference in chunks to reduce memory usage.

        Splits the batch into smaller chunks, processes each independently,
        and concatenates the results.

        Args:
            batch: Full batch dictionary

        Returns:
            Dictionary of model outputs with concatenated chunk results
        """
        batch_size = batch[self.sample_size_property].shape[1]
        chunk_indices = self._get_chunk_indices(batch_size)

        model_outputs = defaultdict(list)
        for start_idx, end_idx in chunk_indices:
            chunked_batch = self._create_chunked_batch(batch, start_idx, end_idx)
            forward_inputs = {k: v for k, v in chunked_batch.items() if k in self.forward_properties}

            with self.trainer.autocast_context:
                chunked_outputs = self.model(**forward_inputs)

            # Accumulate outputs
            for key, value in chunked_outputs.items():
                model_outputs[key].append(value)

        # Concatenate all chunks
        return {key: torch.cat(chunks, dim=1) for key, chunks in model_outputs.items()}

    def _run_model_inference(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Run model inference, optionally in chunks.

        Args:
            batch: Input batch dictionary

        Returns:
            Dictionary of model outputs
        """
        if self.chunked_inference:
            return self._chunked_model_inference(batch)
        else:
            forward_inputs = {k: v for k, v in batch.items() if k in self.forward_properties}
            with self.trainer.autocast_context:
                return self.model(**forward_inputs)

    def _align_chunk_sizes(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Align prediction and target sizes when using chunked inference.

        Args:
            prediction: Prediction tensor
            target: Target tensor

        Returns:
            Tuple of (aligned_prediction, aligned_target)
        """
        if self.chunked_inference and prediction.shape[1] != target.shape[1]:
            min_size = min(prediction.shape[1], target.shape[1])
            prediction = prediction[:, :min_size]
            target = target[:, :min_size]
        return prediction, target

    def _compute_mode_metrics(
        self, batch: dict[str, torch.Tensor], model_outputs: dict[str, torch.Tensor], mode: str
    ) -> dict[str, torch.Tensor]:
        """
        Compute metrics for a specific evaluation mode.

        Args:
            batch: Input batch containing targets
            model_outputs: Model predictions
            mode: Evaluation mode (field name)

        Returns:
            Dictionary of computed metrics for this mode
        """
        target = batch.get(f"{mode}{METRIC_SUFFIX_TARGET}")
        prediction = model_outputs.get(mode)

        if prediction is None or target is None:
            return {}

        # Denormalize
        denorm_pred, denorm_target = self._denormalize(prediction, target, mode)

        # Align sizes if needed
        denorm_pred, denorm_target = self._align_chunk_sizes(denorm_pred, denorm_target)

        # Compute metrics
        return self._compute_metrics(denorm_pred, denorm_target, mode)

    def _forward(self, batch: dict[str, torch.Tensor], **_) -> dict[str, torch.Tensor]:
        """
        Execute forward pass and compute metrics.

        Args:
            batch: Input batch dictionary
            **_: Additional unused arguments

        Returns:
            Dictionary mapping metric names to computed values
        """
        model_outputs = self._run_model_inference(batch)

        metrics = {}
        for mode in self.evaluation_modes:
            metrics.update(self._compute_mode_metrics(batch, model_outputs, mode))

        return metrics

    def _process_results(self, results: dict[str, torch.Tensor], **_) -> None:
        """
        Log computed metrics to writer.

        Args:
            results: Dictionary of computed metrics
            **_: Additional unused arguments
        """
        if not results:
            self.logger.warning(f"No metrics computed for dataset '{self.dataset_key}'")
            return

        for name, metric in results.items():
            metric_key = f"{METRIC_PREFIX_LOSS}{self.dataset_key}/{name}"
            self.writer.add_scalar(
                key=metric_key,
                value=metric.mean(),
                logger=self.logger,
                format_str=".6f",
            )

        self.logger.debug(f"Logged {len(results)} metrics for dataset '{self.dataset_key}'")
