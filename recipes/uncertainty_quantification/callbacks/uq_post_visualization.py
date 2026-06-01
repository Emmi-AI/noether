#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from pydantic import Field, model_validator

from noether.core.callbacks.periodic import PeriodicDataIteratorCallback, PeriodicDataIteratorCallbackConfig

if TYPE_CHECKING:
    import pyvista as pv

VISUALIZATION_FIELDS = ["surface_pressure", "surface_friction"]
METRIC_SUFFIX_TARGET = "_target"
POSITION_NORMALIZER_KEY = "surface_position"

# Same mapping as ``scripts/uq_postprocessing.py``.
SURFACE_GT_MAP = {
    "surface_pressure": "pMeanTrim",
    "surface_friction": "wallShearStressMeanTrim",
}


class UQPostVisualizationCallbackConfig(PeriodicDataIteratorCallbackConfig):
    """Configuration for the UQ post-visualization callback."""

    kind: str | None = "uncertainty_quantification.callbacks.UQPostVisualizationCallback"

    forward_properties: list[str] = []
    """List of properties in the dataset to be forwarded during inference."""
    batch_size: int = Field(1)
    """Batch size for evaluation. Currently only batch_size=1 is supported."""
    chunked_inference: bool = False
    """If True, run model inference in chunks across the sample dimension."""
    chunk_properties: list[str] = []
    """Properties to slice together across the sample dim when chunking."""
    chunk_size: int | None = None
    """Number of points per chunk when ``chunked_inference`` is True."""
    sample_size_property: str | None = None
    """Batch property whose ``shape[1]`` defines the total sample size for chunking."""
    surface_vtp_root: str | None = None
    """Root directory containing ``run_<id>/boundary_<id>.vtp`` files. If ``None``, VTP rendering is skipped."""
    query_chunk_size: int = 16384
    """Chunk size for ``query_surface_position`` when rendering the VTP mesh."""
    anchor_subsample_size: int | None = 16384
    """If set, subsample ``surface_anchor_position`` / ``volume_anchor_position`` to this many points
    when running the query-based VTP inference (matches the training-size batch used in the script)."""

    @model_validator(mode="after")
    def _validate(self) -> UQPostVisualizationCallbackConfig:
        if self.batch_size != 1:
            raise ValueError("UQPostVisualizationCallback only supports batch_size=1")
        if self.chunked_inference:
            if self.chunk_size is None:
                raise ValueError("chunk_size must be specified when chunked_inference is True")
            if not self.chunk_properties:
                raise ValueError("chunk_properties must be specified when chunked_inference is True")
            if self.sample_size_property is None:
                raise ValueError("sample_size_property must be specified when chunked_inference is True")
        return self


class UQPostVisualizationCallback(PeriodicDataIteratorCallback):
    """Render surface predictions + UQ diagnostics, mirroring ``scripts/uq_postprocessing.py``.

    For every sample in the configured dataset:

    * Run anchor-chunked model inference (matches the script's ``chunked_inference``)
    * Aggregate per-element absolute errors and predicted σ across all samples for the
      calibration / error-vs-uncertainty plots (matches ``plot_uq_validity``)
    * If ``surface_vtp_root`` is configured, load the per-sample VTP via
      ``load_surface_mesh``, run ``chunked_query_inference`` at the cell centers,
      attach the denormalized prediction, ground truth, error (+ aleatoric σ for UQ)
      to the mesh, and render the same panel layout as ``render_surface_plots``

    UQ models are detected from the presence of ``{field}_log_var`` outputs.
    """

    def __init__(self, callback_config: UQPostVisualizationCallbackConfig, **kwargs):
        super().__init__(callback_config, **kwargs)

        self._config = callback_config
        self.forward_properties = callback_config.forward_properties
        self.chunked_inference = callback_config.chunked_inference
        self.chunk_properties = callback_config.chunk_properties
        self.chunk_size = callback_config.chunk_size
        self.sample_size_property = callback_config.sample_size_property
        self.surface_vtp_root = Path(callback_config.surface_vtp_root) if callback_config.surface_vtp_root else None
        self.query_chunk_size = callback_config.query_chunk_size
        self.anchor_subsample_size = callback_config.anchor_subsample_size
        self._sample_counter = 0
        self._is_uq = False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _chunked_model_inference(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Chunk ``chunk_properties`` jointly across ``shape[1]`` and concatenate outputs."""
        assert self.sample_size_property is not None and self.chunk_size is not None
        chunk_size = self.chunk_size
        total = batch[self.sample_size_property].shape[1]
        num_chunks = math.ceil(total / chunk_size)

        outputs: dict[str, list[torch.Tensor]] = defaultdict(list)
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, total)
            chunk_batch = {
                key: (value[:, start:end] if key in self.chunk_properties else value) for key, value in batch.items()
            }
            forward_inputs = {k: v for k, v in chunk_batch.items() if k in self.forward_properties}
            with self.trainer.autocast_context:
                chunk_out = self.model(**forward_inputs)
            for key, value in chunk_out.items():
                outputs[key].append(value)

        return {key: torch.cat(chunks, dim=1) for key, chunks in outputs.items()}

    def _run_model_inference(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.chunked_inference:
            return self._chunked_model_inference(batch)
        forward_inputs = {k: v for k, v in batch.items() if k in self.forward_properties}
        with self.trainer.autocast_context:
            outputs: dict[str, torch.Tensor] = self.model(**forward_inputs)
        return outputs

    def _chunked_query_inference(
        self,
        fwd_batch: dict[str, torch.Tensor],
        query_positions: torch.Tensor,
        query_type: str = "surface",
    ) -> dict[str, torch.Tensor]:
        """Query at ``query_positions`` in chunks, keeping geometry + anchors constant.

        Mirrors ``chunked_query_inference`` in ``scripts/uq_postprocessing.py``.
        """
        query_key = f"query_{query_type}_position"
        prefix = f"query_{query_type}_"
        n = query_positions.shape[0]
        chunk_size = self.query_chunk_size
        n_chunks = max(1, (n + chunk_size - 1) // chunk_size)

        outputs: dict[str, list[torch.Tensor]] = defaultdict(list)
        for i in range(n_chunks):
            start, end = i * chunk_size, min((i + 1) * chunk_size, n)
            chunk_batch = dict(fwd_batch)
            chunk_batch[query_key] = query_positions[start:end].unsqueeze(0).to(self.model_device)

            with self.trainer.autocast_context:
                out = self.model(**chunk_batch)

            for key, val in out.items():
                if key.startswith(prefix):
                    clean_key = key.replace(prefix, f"{query_type}_")
                    outputs[clean_key].append(val.cpu().float())

        return {key: torch.cat(chunks, dim=1) for key, chunks in outputs.items()}

    @property
    def model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    # ------------------------------------------------------------------
    # Denormalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _denormalize_std(dataset, mode: str, std_normalized: torch.Tensor) -> torch.Tensor:
        """Denormalize a standard deviation: keep the scale, drop the shift."""
        zero = torch.zeros_like(std_normalized)
        scaled: torch.Tensor = dataset.denormalize(mode, std_normalized) - dataset.denormalize(mode, zero)
        return scaled.abs()

    # ------------------------------------------------------------------
    # Dataset / mesh helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_run_id(dataset, sample_idx: int) -> int | None:
        """Walk wrapper chain to map a wrapped sample index → underlying ``design_ids`` entry."""
        idx = sample_idx
        current = dataset
        while hasattr(current, "dataset") and hasattr(current, "indices"):
            idx = int(current.indices[idx])
            current = current.dataset
        # Bare DatasetWrapper without index remap: just descend.
        while hasattr(current, "dataset") and not hasattr(current, "design_ids"):
            current = current.dataset
        design_ids = getattr(current, "design_ids", None)
        if design_ids is None:
            return None
        return int(design_ids[idx])

    def _load_surface_mesh(self, run_id: int) -> pv.PolyData | None:
        """Mirror ``load_surface_mesh`` in ``scripts/uq_postprocessing.py``."""
        import pyvista as pv

        if self.surface_vtp_root is None:
            return None
        vtp_path = self.surface_vtp_root / f"run_{run_id}" / f"boundary_{run_id}.vtp"
        if not vtp_path.exists():
            self.logger.warning(f"VTP not found: {vtp_path}")
            return None
        return pv.read(str(vtp_path))

    def _build_query_forward_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Build the geometry + (subsampled) anchor batch used for ``chunked_query_inference``."""
        n_anchor = self.anchor_subsample_size
        fwd: dict[str, torch.Tensor] = {}
        for key in self.forward_properties:
            value = batch.get(key)
            if value is None:
                continue
            if n_anchor is not None and key in ("surface_anchor_position", "volume_anchor_position"):
                fwd[key] = value[:, :n_anchor]
            else:
                fwd[key] = value
        return fwd

    # ------------------------------------------------------------------
    # process_data: per-sample inference + rendering
    # ------------------------------------------------------------------

    def process_data(self, batch: dict[str, torch.Tensor], *, trainer_model) -> dict[str, torch.Tensor]:
        predictions = self._run_model_inference(batch)
        is_uq = any(key.endswith("_log_var") for key in predictions)
        self._is_uq = is_uq

        dataset = self.data_container.get_dataset(self.dataset_key)
        sample_idx = self._sample_counter
        self._sample_counter += 1

        result: dict[str, torch.Tensor] = {}

        # ---- Calibration aggregation (matches ``plot_uq_validity``) ----
        for field in VISUALIZATION_FIELDS:
            pred_key = f"{field}_mean" if is_uq else field
            target_key = f"{field}{METRIC_SUFFIX_TARGET}"

            pred = predictions.get(pred_key)
            target = batch.get(target_key)
            if pred is None or target is None:
                continue

            # Chunked inference may truncate to a multiple of chunk_size.
            min_n = min(pred.shape[1], target.shape[1])
            pred = pred[:, :min_n]
            target = target[:, :min_n]

            denorm_pred = dataset.denormalize(field, pred).cpu()
            denorm_target = dataset.denormalize(field, target).cpu()

            # plot_uq_validity uses element-wise abs error (no vector-norm collapse).
            error = (denorm_pred - denorm_target).abs()
            result[f"{field}_errors"] = error.flatten()

            if is_uq:
                logvar = predictions.get(f"{field}_log_var")
                if logvar is not None:
                    std_norm = torch.exp(0.5 * logvar[:, :min_n])
                    std_phys = self._denormalize_std(dataset, field, std_norm).cpu()
                    result[f"{field}_stds"] = std_phys.flatten()

        # ---- Per-sample VTP rendering (matches ``render_surface_plots``) ----
        if self.surface_vtp_root is not None:
            self._render_sample_on_vtp(batch, dataset, sample_idx, is_uq)

        return result

    def _render_sample_on_vtp(
        self,
        batch: dict[str, torch.Tensor],
        dataset,
        sample_idx: int,
        is_uq: bool,
    ) -> None:
        run_id = self._resolve_run_id(dataset, sample_idx)
        if run_id is None:
            self.logger.warning(f"Could not resolve run_id for sample {sample_idx}; skipping VTP rendering")
            return

        mesh = self._load_surface_mesh(run_id)
        if mesh is None:
            return

        self.logger.info(f"Full-mesh query inference for sample {sample_idx} (run_{run_id}, {mesh.n_cells} cells)")

        cell_centers = torch.tensor(mesh.cell_centers().points, dtype=torch.float32)
        # Normalize cell centers using the dataset's position normalizer (script: pos_normalizer(cell_centers)).
        pos_normalizer = dataset.normalizers.get(POSITION_NORMALIZER_KEY)
        cell_centers_norm = pos_normalizer(cell_centers) if pos_normalizer is not None else cell_centers

        fwd_batch = self._build_query_forward_batch(batch)
        vtp_preds = self._chunked_query_inference(fwd_batch, cell_centers_norm, query_type="surface")

        self._attach_mesh_predictions(mesh, vtp_preds, dataset, is_uq)
        self._render_and_save_mesh(mesh, sample_idx, is_uq)

    def _attach_mesh_predictions(
        self,
        mesh: pv.PolyData,
        vtp_preds: dict[str, torch.Tensor],
        dataset,
        is_uq: bool,
    ) -> None:
        for field, gt_key in SURFACE_GT_MAP.items():
            if gt_key not in mesh.cell_data:
                continue
            gt = mesh.cell_data[gt_key]
            field_short = field.replace("surface_", "")

            pred_key = f"{field}_mean" if is_uq else field
            pred_norm = vtp_preds.get(pred_key)
            if pred_norm is None:
                continue

            denorm_pred = dataset.denormalize(field, pred_norm).cpu()[0].numpy().squeeze()

            if denorm_pred.ndim == 1 or (denorm_pred.ndim == 2 and denorm_pred.shape[-1] == 1):
                denorm_pred = denorm_pred.squeeze()
                mesh.cell_data[f"pred_{field_short}"] = denorm_pred
                mesh.cell_data[f"error_{field_short}"] = np.abs(denorm_pred - gt.squeeze())
            else:
                mesh.cell_data[f"pred_{field_short}_mag"] = np.linalg.norm(denorm_pred, axis=-1)
                mesh.cell_data[f"gt_{field_short}_mag"] = np.linalg.norm(gt, axis=-1)
                mesh.cell_data[f"error_{field_short}"] = np.linalg.norm(denorm_pred - gt, axis=-1)

            if is_uq:
                logvar = vtp_preds.get(f"{field}_log_var")
                if logvar is not None:
                    std_denorm = self._denormalize_std(dataset, field, torch.exp(0.5 * logvar))
                    std_np = std_denorm[0].numpy()
                    if std_np.ndim > 1 and std_np.shape[-1] > 1:
                        std_np = np.linalg.norm(std_np, axis=-1)
                    mesh.cell_data[f"aleatoric_std_{field_short}"] = std_np.squeeze()

    def _render_and_save_mesh(self, mesh: pv.PolyData, sample_idx: int, is_uq: bool) -> None:
        """Render the per-field panel layout from ``render_surface_plots`` and save as PNG."""
        import matplotlib.pyplot as plt
        import pyvista as pv

        pv.OFF_SCREEN = True
        cp = self.trainer.update_counter.cur_iteration
        out_dir = self.checkpoint_writer.path_provider.run_output_path / "uq_visualization" / self.dataset_key
        out_dir.mkdir(parents=True, exist_ok=True)

        for field, gt_key in SURFACE_GT_MAP.items():
            field_short = field.replace("surface_", "")

            gt_plot_key = f"gt_{field_short}_mag" if f"gt_{field_short}_mag" in mesh.cell_data else gt_key
            pred_plot_key = (
                f"pred_{field_short}_mag" if f"pred_{field_short}_mag" in mesh.cell_data else f"pred_{field_short}"
            )
            error_key = f"error_{field_short}"
            ale_key = f"aleatoric_std_{field_short}"

            shared_clim = None
            if gt_plot_key in mesh.cell_data:
                gt_data = mesh.cell_data[gt_plot_key]
                shared_clim = [float(np.percentile(gt_data, 1)), float(np.percentile(gt_data, 99))]

            panels: list[tuple[str, str, str, list[float] | None]] = []
            if gt_plot_key in mesh.cell_data:
                panels.append(("Ground Truth", gt_plot_key, "coolwarm", shared_clim))
            if pred_plot_key in mesh.cell_data:
                panels.append(("Prediction", pred_plot_key, "coolwarm", shared_clim))
            if error_key in mesh.cell_data:
                err_data = mesh.cell_data[error_key]
                panels.append(("|Error|", error_key, "Reds", [0.0, float(np.percentile(err_data, 95))]))
            if is_uq and ale_key in mesh.cell_data:
                ale_data = mesh.cell_data[ale_key]
                panels.append(
                    (
                        "Aleatoric σ",
                        ale_key,
                        "Reds",
                        [float(np.percentile(ale_data, 5)), float(np.percentile(ale_data, 95))],
                    )
                )

            if not panels:
                continue

            panel_images: list[np.ndarray] = []
            for title, array_name, cmap, clim in panels:
                plotter = pv.Plotter(off_screen=True, window_size=[600, 500])
                plotter.add_mesh(
                    mesh.copy(),
                    scalars=array_name,
                    cmap=cmap,
                    clim=clim,
                    show_scalar_bar=True,
                    scalar_bar_args={"title": title, "n_labels": 5},
                )
                plotter.add_text(title, font_size=12, position="upper_left")
                plotter.camera_position = "xy"
                plotter.camera.zoom(1.5)
                panel_images.append(plotter.screenshot(return_img=True))
                plotter.close()

            fig, axes = plt.subplots(1, len(panel_images), figsize=(6 * len(panel_images), 5))
            if len(panel_images) == 1:
                axes = [axes]
            for ax, img in zip(axes, panel_images, strict=True):
                ax.imshow(img)
                ax.axis("off")
            plt.tight_layout(pad=0.5)

            img_path = out_dir / f"{field_short}_sample{sample_idx}_cp={cp}.png"
            fig.savefig(img_path, dpi=150, bbox_inches="tight")
            self.writer.add_nonscalar(
                key=f"uq_visualization/{self.dataset_key}/sample_{sample_idx:04d}/{field_short}",
                value=fig,
            )
            plt.close(fig)

    # ------------------------------------------------------------------
    # process_results: calibration + error-vs-uncertainty aggregates
    # ------------------------------------------------------------------

    def process_results(self, results: dict[str, torch.Tensor], *, interval_type, update_counter, **_) -> None:
        try:
            if not results:
                self.logger.warning(f"No results for dataset '{self.dataset_key}'")
                return

            if not self._is_uq:
                self.logger.debug(
                    f"Skipping UQ aggregate plots for dataset '{self.dataset_key}': model has no log_var outputs"
                )
                return

            for field in VISUALIZATION_FIELDS:
                errors_key = f"{field}_errors"
                stds_key = f"{field}_stds"
                if errors_key not in results or stds_key not in results:
                    continue

                errors = results[errors_key].cpu().numpy()
                stds = results[stds_key].cpu().numpy()
                if errors.size == 0 or stds.size == 0:
                    continue

                field_short = field.replace("surface_", "")
                self._log_calibration(field_short, errors, stds)
                self._log_error_vs_uncertainty(field_short, errors, stds)
        finally:
            self._sample_counter = 0
            self._is_uq = False

    def _log_calibration(self, field_short: str, errors: np.ndarray, stds: np.ndarray) -> None:
        from scipy.stats import norm as scipy_norm

        confidence_levels = np.linspace(0.05, 0.95, 19)
        z_scores = np.array([scipy_norm.ppf(0.5 + c / 2) for c in confidence_levels])
        coverages = np.array([(errors < z * stds).mean() for z in z_scores])

        fig = self._calibration_figure(field_short, confidence_levels, coverages)
        self._write_aggregate_figure(field_short, "calibration", fig)

    def _log_error_vs_uncertainty(self, field_short: str, errors: np.ndarray, stds: np.ndarray) -> None:
        n_bins = 30
        bins = np.percentile(stds, np.linspace(0, 100, n_bins + 1))
        bin_centers: list[float] = []
        bin_errors: list[float] = []
        for k in range(n_bins):
            mask = (stds >= bins[k]) & (stds < bins[k + 1])
            if mask.sum() > 100:
                bin_centers.append(float(stds[mask].mean()))
                bin_errors.append(float(errors[mask].mean()))

        if not bin_centers:
            return

        fig = self._error_vs_uncertainty_figure(field_short, bin_centers, bin_errors)
        self._write_aggregate_figure(field_short, "error_vs_uncertainty", fig)

    def _write_aggregate_figure(self, field_short: str, plot_name: str, fig) -> None:
        import matplotlib.pyplot as plt

        cp = self.trainer.update_counter.cur_iteration
        out_dir = self.checkpoint_writer.path_provider.run_output_path / "uq_visualization" / self.dataset_key
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{plot_name}_{field_short}_cp={cp}.png", dpi=150, bbox_inches="tight")
        self.writer.add_nonscalar(
            key=f"uq_visualization/{self.dataset_key}/{plot_name}/{field_short}",
            value=fig,
        )
        plt.close(fig)

    # ------------------------------------------------------------------
    # Aggregate figures (matplotlib — match ``plot_uq_validity``)
    # ------------------------------------------------------------------

    @staticmethod
    def _calibration_figure(field_short: str, confidence_levels: np.ndarray, coverages: np.ndarray):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(confidence_levels, coverages, "b-o", markersize=4, label="Observed")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.fill_between(confidence_levels, coverages, confidence_levels, alpha=0.15, color="blue")
        ax.set_xlabel("Expected coverage")
        ax.set_ylabel("Observed coverage")
        ax.set_title(f"Calibration: {field_short}")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        plt.tight_layout()
        return fig

    @staticmethod
    def _error_vs_uncertainty_figure(field_short: str, bin_centers: list[float], bin_errors: list[float]):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(bin_centers, bin_errors, s=30, edgecolors="k", linewidths=0.5)
        lim = max(*bin_centers, *bin_errors) * 1.1
        ax.plot([0, lim], [0, lim], "r--", lw=2, label="y = x (perfect)")
        ax.set_xlabel("Predicted σ (binned)")
        ax.set_ylabel("Mean |error|")
        ax.set_title(f"Error vs Uncertainty: {field_short}")
        ax.legend()
        plt.tight_layout()
        return fig
