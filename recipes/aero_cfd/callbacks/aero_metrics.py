#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import torch
from pydantic import Field, model_validator

from noether.core.callbacks.periodic import PeriodicDataIteratorCallback
from noether.core.schemas.callbacks import PeriodicDataIteratorCallbackConfig


class AeroMetricsCallbackConfig(PeriodicDataIteratorCallbackConfig):
    """Configuration for surface/volume evaluation metrics callback."""

    kind: str | None = "aero_cfd.callbacks.AeroMetricsCallback"

    forward_properties: list[str] = []
    """List of properties in the dataset to be forwarded during inference."""
    chunked_inference: bool = False
    """If True, perform inference in chunks over the full simulation geometry."""
    chunk_properties: list[str] = []
    """List of properties in the dataset to be chunked for chunked inference."""
    batch_size: int = Field(1)
    """Batch size for evaluation. Currently only batch_size=1 is supported."""
    chunk_size: int | None = None
    """Size of each chunk when performing chunked inference."""
    sample_size_property: str | None = Field(None)
    """Property in the batch to determine the sample size for chunking."""
    save_predictions: bool = False
    """If True, save denormalized predictions to disk during evaluation."""
    predictions_path: str | None = None
    """Directory to save per-sample prediction files. Required when save_predictions=True."""
    batch_properties_to_save: list[str] = []
    """Batch keys (e.g. position tensors) to save alongside predictions."""
    compute_forces: bool = False
    """If True, compute drag/lift coefficients per sample and log errors."""

    @model_validator(mode="after")
    def validate_config(self) -> AeroMetricsCallbackConfig:
        if self.batch_size != 1:
            raise ValueError("AeroMetricsCallback only supports batch_size=1")
        if self.save_predictions and self.predictions_path is None:
            raise ValueError("predictions_path must be specified when save_predictions=True")
        if self.chunked_inference:
            if self.chunk_size is None:
                raise ValueError("chunk_size must be specified when chunked_inference is True")
            if not self.forward_properties:
                raise ValueError("forward_properties must be specified when chunked_inference is True")
            if not self.chunk_properties:
                raise ValueError("chunk_properties must be specified when chunked_inference is True")
        return self


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


class AeroMetricsCallback(PeriodicDataIteratorCallback):
    """Evaluation callback for aerodynamic surface and volume predictions.

    Computes MSE, MAE, and relative L2 error metrics for physical fields
    (pressure, friction, velocity, vorticity) by running model inference on
    an evaluation dataset.  Supports chunked inference for memory efficiency.

    When ``save_predictions=True``, denormalized predictions (and optionally
    batch properties such as positions) are saved to disk per-sample for
    downstream use (VTK export, force coefficient computation).

    Args:
        callback_config: Configuration for the callback including dataset key,
            forward properties, and chunking settings.
        **kwargs: Additional arguments passed to parent class.

    Attributes:
        dataset_key: Identifier for the dataset to evaluate.
        evaluation_modes: List of field names to evaluate.
        dataset_normalizers: Normalizers for denormalizing predictions.
        forward_properties: Properties to pass to model forward.
        chunked_inference: Whether to use chunked inference.
        chunk_properties: Properties to chunk.
        chunk_size: Size of each chunk.
        sample_size_property: Property to determine chunk count.
    """

    def __init__(self, callback_config: AeroMetricsCallbackConfig, **kwargs):
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
        self._save_predictions = callback_config.save_predictions
        self._predictions_path = callback_config.predictions_path
        self._prediction_counter: int = 0
        self._compute_forces = callback_config.compute_forces
        if self._compute_forces:
            from scipy.spatial import cKDTree

            from aero_cfd.utils.drag_lift import FlowConditions, compute_force_coefficients

            self._cKDTree = cKDTree
            self._FlowConditions = FlowConditions
            self._compute_force_coefficients = compute_force_coefficients

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
        num_chunks = math.ceil(batch_size / self.chunk_size)

        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, batch_size)
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

        dataset = self.data_container.get_dataset(self.dataset_key)
        denorm_pred = dataset.denormalize(mode, prediction)
        denorm_target = dataset.denormalize(mode, target)

        # Align sizes if needed
        denorm_pred, denorm_target = self._align_chunk_sizes(denorm_pred, denorm_target)

        # Compute metrics
        return self._compute_metrics(denorm_pred, denorm_target, mode)

    def _compute_force_metrics(
        self, batch: dict[str, torch.Tensor], model_outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Compute drag/lift coefficient errors for the current sample.

        Uses full-resolution mesh geometry from the batch (``surface_normals``,
        ``surface_area``, ``surface_position``) and loads full-resolution GT
        fields from disk (since batch targets are subsampled by the pipeline).
        Predicted Cd/Cl uses denormalized model outputs matched to the mesh via
        nearest-neighbor lookup.

        Requires ``surface_normals``, ``surface_area``, and ``surface_position``
        to be present in the batch. Enable these by removing them from
        ``excluded_properties`` in the dataset config.
        """
        # Full-resolution mesh geometry from batch
        surface_normals = batch.get("surface_normals")
        surface_areas = batch.get("surface_area")
        mesh_positions = batch.get("surface_position")

        if surface_normals is None or surface_areas is None or mesh_positions is None:
            self.logger.warning(
                "Skipping force computation: surface_normals, surface_area, or surface_position "
                "not in batch. Ensure these fields are not excluded in the dataset config."
            )
            return {}

        surface_normals = surface_normals.cpu().squeeze(0).float()
        surface_areas = surface_areas.cpu().squeeze(0).float()
        mesh_positions = mesh_positions.cpu().squeeze(0).float()

        # Ground-truth Cd/Cl from full-resolution dataset files.
        # Batch targets are subsampled by the pipeline, so we load the originals.
        dataset = self.data_container.get_dataset(self.dataset_key)
        sample_idx = batch["index"].squeeze().item()
        info = dataset.sample_info(sample_idx)
        run_dir = Path(info["sample_uri"])

        # Load per-run reference area if available, otherwise use defaults.
        design_id = info["design_id"]
        ref_csv = run_dir / f"geo_ref_{design_id}.csv"
        if ref_csv.exists():
            import pandas as pd

            ref_area = float(pd.read_csv(ref_csv)["aRef"][0])
            flow = self._FlowConditions(reference_area=ref_area)
        else:
            flow = self._FlowConditions()

        gt_pressure_path = run_dir / "surface_pressure.pt"
        gt_shear_path = run_dir / "surface_wallshearstress.pt"
        if not gt_pressure_path.exists() or not gt_shear_path.exists():
            self.logger.debug(f"Skipping GT force computation for sample {sample_idx}: missing GT files")
            return {}

        gt_pressure = torch.load(gt_pressure_path, map_location="cpu", weights_only=True).float()
        gt_shear = torch.load(gt_shear_path, map_location="cpu", weights_only=True).float()
        if gt_pressure.ndim == 2 and gt_pressure.shape[-1] == 1:
            gt_pressure = gt_pressure.squeeze(-1)

        gt_coeffs = self._compute_force_coefficients(gt_pressure, gt_shear, surface_normals, surface_areas, flow)

        # Predicted Cd/Cl from model outputs (denormalized)
        pred_pressure = model_outputs.get("surface_pressure")
        pred_friction = model_outputs.get("surface_friction")
        pred_positions = batch.get("surface_anchor_position")

        if pred_pressure is None or pred_friction is None or pred_positions is None:
            return {}

        pred_pressure_denorm = self.dataset_normalizers["surface_pressure"].inverse(pred_pressure.cpu()).squeeze(0)
        pred_friction_denorm = self.dataset_normalizers["surface_friction"].inverse(pred_friction.cpu()).squeeze(0)
        pred_positions_cpu = pred_positions.cpu().squeeze(0)

        if pred_pressure_denorm.ndim == 2 and pred_pressure_denorm.shape[-1] == 1:
            pred_pressure_denorm = pred_pressure_denorm.squeeze(-1)

        # Match predicted positions to mesh positions for normals/areas lookup
        position_tree = self._cKDTree(mesh_positions.numpy())
        _, matched_indices = position_tree.query(pred_positions_cpu.numpy())

        pred_coeffs = self._compute_force_coefficients(
            pred_pressure_denorm,
            pred_friction_denorm,
            surface_normals[matched_indices],
            surface_areas[matched_indices],
            flow,
        )

        return {
            "drag_error": torch.tensor(abs(gt_coeffs.cd - pred_coeffs.cd)),
            "lift_error": torch.tensor(abs(gt_coeffs.cl - pred_coeffs.cl)),
        }

    def process_data(self, batch: dict[str, torch.Tensor], **_) -> dict[str, torch.Tensor]:
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

        if self._compute_forces:
            metrics.update(self._compute_force_metrics(batch, model_outputs))

        if self._save_predictions:
            self._collect_predictions(batch, model_outputs)

        return metrics

    def _collect_predictions(self, batch: dict[str, torch.Tensor], model_outputs: dict[str, torch.Tensor]) -> None:
        """Denormalize and save predictions (and batch properties) for the current sample.

        Saves each sample to disk immediately to avoid accumulating large tensors in memory.
        """
        sample = {}
        for mode in self.evaluation_modes:
            prediction = model_outputs.get(mode)
            if prediction is None:
                continue
            normalizer = self.dataset_normalizers.get(mode)
            if normalizer is not None:
                denorm = normalizer.inverse(prediction.cpu())
            else:
                denorm = prediction.cpu()
            sample[mode] = denorm.squeeze(0)
        for key in self._config.batch_properties_to_save:
            if key in batch:
                sample[key] = batch[key].cpu().squeeze(0)
        if sample:
            out_dir = Path(self._predictions_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            idx = self._prediction_counter
            torch.save(sample, out_dir / f"sample_{idx:04d}.pt")
            self._prediction_counter += 1

    def process_results(self, results: dict[str, torch.Tensor], **_) -> None:
        """
        Log computed metrics to writer and optionally save predictions.

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

        if self._save_predictions and self._prediction_counter > 0:
            self.logger.info(f"Saved {self._prediction_counter} prediction files to {self._predictions_path}")
            self._prediction_counter = 0
