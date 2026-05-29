# ruff: noqa: B905, PLW3301
#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Post-processing script for AB-UPT models (baseline and UQ).

Replicates the SurfaceVolumeEvaluationMetricsCallback chunked inference exactly:
- Builds inference pipeline (all points) to get batch with positions + targets
- Chunks surface_anchor_position and volume_anchor_position together
- Denormalizes both predictions and targets using dataset normalizers
- Computes MSE, MAE, relative L2 in physical units
- Renders surface predictions on VTP mesh with pyvista

Usage:
    uv run python recipes/uncertainty_quantification/scripts/uq_postprocessing.py \
        --run-dir outputs/<baseline> outputs/<uq_run> \
        --labels baseline uq_default --num-samples 3
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
import yaml

from noether.core.factory import Factory
from noether.core.factory.dataset import DatasetFactory
from noether.core.factory.utils import class_constructor_from_class_path
from noether.core.schemas.lib import resolve_config_class
from noether.core.schemas.models import ModelBaseConfig

SURFACE_VTP_ROOT = Path("/nfs-gpu/research/datasets/drivaerml/raw_surface_data")
CHUNK_SIZE = 16384
CHUNK_PROPERTIES = ["surface_anchor_position", "volume_anchor_position"]
FORWARD_PROPERTIES = [
    "geometry_position",
    "geometry_supernode_idx",
    "geometry_batch_idx",
    "surface_anchor_position",
    "volume_anchor_position",
]
EVALUATION_MODES = [
    "surface_pressure",
    "surface_friction",
    "volume_velocity",
    "volume_pressure",
    "volume_vorticity",
]
SURFACE_GT_MAP = {
    "surface_pressure": "pMeanTrim",
    "surface_friction": "wallShearStressMeanTrim",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_model_and_data(run_dir: Path, checkpoint: str, device: str):
    """Load model, dataset with inference pipeline (all points), and normalizers."""
    stage_dir = run_dir / "train"
    with open(stage_dir / "hp_resolved.yaml") as f:
        config = yaml.full_load(f)

    # Checkpoint
    ckpt_dir = stage_dir / "checkpoints"
    if checkpoint == "best":
        ckpt_files = list(ckpt_dir.glob("*best*model.th"))
        if not ckpt_files:
            checkpoint = "latest"
        else:
            ckpt_path = ckpt_files[0]
    if checkpoint == "latest":
        ckpt_path = next(ckpt_dir.glob("*latest_model.th"))
    if checkpoint not in ("best", "latest"):
        ckpt_path = ckpt_dir / checkpoint
    print(f"  Checkpoint: {ckpt_path.name}")

    # Config schema
    config_schema_cls = class_constructor_from_class_path(
        config.get("config_schema_kind", "noether.core.schemas.schema.ConfigSchema")
    )
    model_kind = config["model"].get("kind", "")
    model_config_cls = resolve_config_class(model_kind, ModelBaseConfig)
    computed = set()
    for parent in model_config_cls.__mro__:
        if hasattr(parent, "model_computed_fields"):
            computed |= set(parent.model_computed_fields.keys())
    config["model"] = {k: v for k, v in config["model"].items() if k not in computed}
    validated_config = config_schema_cls(**config)

    # Model
    model = Factory().instantiate(validated_config.model)
    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt_data["state_dict"] if "state_dict" in ckpt_data else ckpt_data

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Dataset with INFERENCE pipeline (all points, like chunked_test)
    test_config = validated_config.datasets["test"]
    dataset = DatasetFactory().instantiate(test_config)

    # Build inference pipeline: same as training but with huge anchor counts
    pipeline_dict = copy.deepcopy(config["datasets"]["test"]["pipeline"])
    pipeline_dict["num_surface_anchor_points"] = 1_000_000_000
    pipeline_dict["num_volume_anchor_points"] = 1_000_000_000
    from aero_cfd.pipeline import AeroCFDPipelineConfig

    pipeline_config = AeroCFDPipelineConfig(**pipeline_dict)
    pipeline_cls = class_constructor_from_class_path(pipeline_dict["kind"])
    inference_pipeline = pipeline_cls(pipeline_config)
    dataset.pipeline = inference_pipeline

    # Dataset normalizers for denormalization
    inner = dataset
    while hasattr(inner, "_dataset"):
        inner = inner._dataset
    normalizers = inner.normalizers if hasattr(inner, "normalizers") else {}

    # Position normalizer (for VTP cell centers)
    pos_normalizer = normalizers.get("surface_position")

    # Also build training pipeline (small anchors) for VTP query inference
    if test_config.pipeline is not None:
        train_pipeline = Factory().create(test_config.pipeline)
    else:
        from noether.data.pipeline import MultiStagePipeline

        train_pipeline = MultiStagePipeline()

    is_uq = hasattr(model, "forward_with_epistemic")
    return model, dataset, inference_pipeline, train_pipeline, config, is_uq, normalizers, pos_normalizer


def get_test_run_ids(dataset) -> list[int]:
    inner = dataset
    while hasattr(inner, "_dataset"):
        inner = inner._dataset
    return sorted(inner.get_dataset_splits.test)


# ---------------------------------------------------------------------------
# Chunked inference (matching callback exactly)
# ---------------------------------------------------------------------------


def chunked_query_inference(
    model,
    fwd_batch: dict,
    query_positions: torch.Tensor,
    device: str,
    query_type: str = "surface",
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, torch.Tensor]:
    """Query at arbitrary positions using AB-UPT's query mechanism.

    Keeps geometry + anchors constant, chunks query_surface_position or
    query_volume_position. Collects only the query outputs.
    """
    query_key = f"query_{query_type}_position"
    n = query_positions.shape[0]
    n_chunks = max(1, (n + chunk_size - 1) // chunk_size)
    outputs: dict[str, list] = defaultdict(list)

    for i in range(n_chunks):
        start, end = i * chunk_size, min((i + 1) * chunk_size, n)
        chunk_batch = dict(fwd_batch)
        chunk_batch[query_key] = query_positions[start:end].unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device != "cpu"):
            out = model(**chunk_batch)

        prefix = f"query_{query_type}_"
        for key, val in out.items():
            if key.startswith(prefix):
                clean_key = key.replace(f"query_{query_type}_", f"{query_type}_")
                outputs[clean_key].append(val.cpu().float())

        if (i + 1) % 50 == 0 or i == n_chunks - 1:
            print(f"    Query chunk {i + 1}/{n_chunks}", end="\r")

    print()
    return {key: torch.cat(chunks, dim=1) for key, chunks in outputs.items()}


def chunked_inference(
    model,
    batch: dict,
    device: str,
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, torch.Tensor]:
    """Chunked inference matching SurfaceVolumeEvaluationMetricsCallback.

    Chunks both surface_anchor_position and volume_anchor_position with the
    same indices, keeping geometry inputs constant.
    """
    sample_size = batch["surface_anchor_position"].shape[1]
    n_chunks = sample_size // chunk_size

    all_outputs: dict[str, list] = defaultdict(list)

    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size

        # Chunk the chunk_properties, keep everything else
        chunk_batch = {}
        for key, value in batch.items():
            if key in CHUNK_PROPERTIES:
                chunk_batch[key] = value[:, start:end]
            else:
                chunk_batch[key] = value

        forward_inputs = {k: v for k, v in chunk_batch.items() if k in FORWARD_PROPERTIES}

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device != "cpu"):
            out = model(**forward_inputs)

        for key, val in out.items():
            all_outputs[key].append(val.cpu().float())

        if (i + 1) % 50 == 0 or i == n_chunks - 1:
            print(f"    Chunk {i + 1}/{n_chunks}", end="\r")

    print()
    return {key: torch.cat(chunks, dim=1) for key, chunks in all_outputs.items()}


# ---------------------------------------------------------------------------
# Denormalization
# ---------------------------------------------------------------------------


def denormalize_field(tensor: torch.Tensor, field: str, normalizers: dict) -> torch.Tensor:
    """Denormalize a tensor using the dataset normalizer (same as callback)."""
    if field not in normalizers:
        return tensor
    return normalizers[field].inverse(tensor.cpu())


def denormalize_std(std_tensor: torch.Tensor, field: str, normalizers: dict) -> torch.Tensor:
    """Denormalize a standard deviation (scale only, no shift).

    inverse() does: x / scale - shift = x * field_std + field_mean
    For a std we only want the scaling: std_phys = std_norm * field_std = std_norm / scale
    So: denorm_std = inverse(std) - inverse(0)
    """
    if field not in normalizers:
        return std_tensor
    n = normalizers[field]
    zero = torch.zeros_like(std_tensor)
    return (n.inverse(std_tensor.cpu()) - n.inverse(zero)).abs()


# ---------------------------------------------------------------------------
# Metrics (matching callback)
# ---------------------------------------------------------------------------


def compute_field_metrics(denorm_pred: torch.Tensor, denorm_target: torch.Tensor, name: str) -> dict[str, float]:
    """MSE, MAE, relative L2 (matching callback _compute_metrics)."""
    delta = denorm_pred - denorm_target
    metrics = {
        f"{name}_mse": float((delta**2).mean()),
        f"{name}_mae": float(delta.abs().mean()),
    }
    target_norm = float(denorm_target.norm())
    if target_norm > 1e-8:
        metrics[f"{name}_l2_rel"] = float(delta.norm() / target_norm)
    return metrics


# ---------------------------------------------------------------------------
# Surface VTP visualization
# ---------------------------------------------------------------------------


def render_surface_plots(mesh: pv.PolyData, output_path: Path, sample_idx: int, is_uq: bool):
    """Render surface colormaps on VTP mesh with pyvista (offscreen)."""
    pv.OFF_SCREEN = True

    for field, gt_key in SURFACE_GT_MAP.items():
        field_short = field.replace("surface_", "")

        gt_plot_key = f"gt_{field_short}_mag" if f"gt_{field_short}_mag" in mesh.cell_data else gt_key
        pred_plot_key = (
            f"pred_{field_short}_mag" if f"pred_{field_short}_mag" in mesh.cell_data else f"pred_{field_short}"
        )
        error_key = f"error_{field_short}"
        ale_key = f"aleatoric_std_{field_short}"

        # Shared clim from GT
        shared_clim = None
        if gt_plot_key in mesh.cell_data:
            gt_data = mesh.cell_data[gt_plot_key]
            shared_clim = [float(np.percentile(gt_data, 1)), float(np.percentile(gt_data, 99))]

        panels = []
        if gt_plot_key in mesh.cell_data:
            panels.append(("Ground Truth", gt_plot_key, "coolwarm", shared_clim))
        if pred_plot_key in mesh.cell_data:
            panels.append(("Prediction", pred_plot_key, "coolwarm", shared_clim))
        if error_key in mesh.cell_data:
            err_data = mesh.cell_data[error_key]
            panels.append(("|Error|", error_key, "Reds", [0, float(np.percentile(err_data, 95))]))
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

        panel_images = []
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
        for ax, img in zip(axes, panel_images):
            ax.imshow(img)
            ax.axis("off")
        plt.tight_layout(pad=0.5)
        img_path = output_path / f"{field_short}_sample{sample_idx}.png"
        fig.savefig(img_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Rendered: {img_path.name}")


# ---------------------------------------------------------------------------
# Summary plots
# ---------------------------------------------------------------------------


def plot_comparison(all_run_metrics: dict[str, list[dict]], output_path: Path):
    """Per-field subplots comparing metrics across runs."""
    all_fields = set()
    for metrics_list in all_run_metrics.values():
        for m in metrics_list:
            all_fields |= {k.replace("_mse", "") for k in m if k.endswith("_mse")}
    fields_present = sorted(all_fields)
    if not fields_present:
        return

    for metric_suffix, metric_label in [("_mse", "MSE"), ("_mae", "MAE"), ("_l2_rel", "Relative L2")]:
        n_fields = len(fields_present)
        fig, axes = plt.subplots(1, n_fields, figsize=(3 * n_fields, 4), sharey=False)
        if n_fields == 1:
            axes = [axes]

        labels = list(all_run_metrics.keys())
        x = np.arange(len(labels))

        for ax, field in zip(axes, fields_present):
            means, stds = [], []
            for label in labels:
                vals = [m.get(f"{field}{metric_suffix}", np.nan) for m in all_run_metrics[label]]
                means.append(np.nanmean(vals))
                stds.append(np.nanstd(vals))
            ax.bar(x, means, yerr=stds, capsize=3, color=plt.cm.Set2(np.linspace(0, 1, len(labels))))
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_title(field, fontsize=10)
            ax.ticklabel_format(axis="y", style="scientific", scilimits=(-2, 3))

        fig.suptitle(f"{metric_label} Comparison (physical units)", fontsize=13)
        plt.tight_layout()
        fname = f"comparison{metric_suffix}.png"
        fig.savefig(output_path / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


def plot_uq_validity(all_run_data: dict[str, dict], output_path: Path):
    """Calibration + error-vs-uncertainty for UQ runs using pipeline predictions/targets."""
    from scipy.stats import norm as scipy_norm

    for label, data in all_run_data.items():
        if not data.get("is_uq"):
            continue

        run_dir = output_path / data["run_name"]
        preds_list = data.get("preds_list", [])
        targets_list = data.get("targets_list", [])
        normalizers = data.get("normalizers", {})

        for field in ["surface_pressure", "surface_friction"]:
            field_short = field.replace("surface_", "")
            mean_key = f"{field}_mean"
            logvar_key = f"{field}_log_var"
            target_key = f"{field}_target"

            all_errors, all_stds = [], []
            for preds, targets in zip(preds_list, targets_list):
                if mean_key not in preds or logvar_key not in preds or target_key not in targets:
                    continue

                # Denormalize
                min_n = min(preds[mean_key].shape[1], targets[target_key].shape[1])
                denorm_pred = denormalize_field(preds[mean_key][:, :min_n], field, normalizers)
                denorm_target = denormalize_field(targets[target_key][:, :min_n], field, normalizers)

                # Denormalize std (scale only, no shift)
                std_norm = torch.exp(0.5 * preds[logvar_key][:, :min_n])
                std_phys = denormalize_std(std_norm, field, normalizers)

                error = (denorm_pred - denorm_target).abs()

                # Flatten
                all_errors.append(error.flatten())
                all_stds.append(std_phys.flatten())

            if not all_errors:
                continue

            errors = torch.cat(all_errors).numpy()
            stds = torch.cat(all_stds).numpy()

            # --- Calibration ---
            confidence_levels = np.linspace(0.05, 0.95, 19)
            z_scores = [scipy_norm.ppf(0.5 + c / 2) for c in confidence_levels]
            coverages = [(errors < z * stds).mean() for z in z_scores]

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(confidence_levels, coverages, "b-o", markersize=4, label="Observed")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
            ax.fill_between(confidence_levels, coverages, confidence_levels, alpha=0.15, color="blue")
            ax.set_xlabel("Expected coverage")
            ax.set_ylabel("Observed coverage")
            ax.set_title(f"Calibration: {field_short} ({label})")
            ax.legend()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            plt.tight_layout()
            fig.savefig(run_dir / f"calibration_{field_short}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: calibration_{field_short}.png")

            # --- Error vs uncertainty (1:1) ---
            n_bins = 30
            bins = np.percentile(stds, np.linspace(0, 100, n_bins + 1))
            bin_centers, bin_errors = [], []
            for k in range(n_bins):
                mask = (stds >= bins[k]) & (stds < bins[k + 1])
                if mask.sum() > 100:
                    bin_centers.append(stds[mask].mean())
                    bin_errors.append(errors[mask].mean())

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(bin_centers, bin_errors, s=30, edgecolors="k", linewidths=0.5)
            if bin_centers:
                lim = max(max(bin_centers), max(bin_errors)) * 1.1
                ax.plot([0, lim], [0, lim], "r--", lw=2, label="y = x (perfect)")
            ax.set_xlabel("Predicted σ (binned)")
            ax.set_ylabel("Mean |error|")
            ax.set_title(f"Error vs Uncertainty: {field_short} ({label})")
            ax.legend()
            plt.tight_layout()
            fig.savefig(run_dir / f"error_vs_uncertainty_{field_short}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: error_vs_uncertainty_{field_short}.png")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate_run(
    run_dir: Path,
    checkpoint: str,
    num_samples: int,
    device: str,
    output_dir: Path,
    save_vtp: bool = False,
) -> tuple[list[dict], dict]:
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {run_dir.name}")
    print(f"{'=' * 60}")

    model, dataset, pipeline, train_pipeline, config, is_uq, normalizers, pos_normalizer = load_model_and_data(
        run_dir, checkpoint, device
    )
    print(f"  Model: {'UQ' if is_uq else 'Baseline'} | Test samples: {len(dataset)}")

    run_output = output_dir / run_dir.name
    run_output.mkdir(parents=True, exist_ok=True)

    test_run_ids = get_test_run_ids(dataset)
    all_metrics = []
    all_preds = []
    all_targets = []
    n = min(num_samples, len(dataset))

    for i in range(n):
        print(f"\n  Sample {i + 1}/{n} (run_{test_run_ids[i]})")

        # Build batch with inference pipeline (ALL points)
        sample = dataset[i]
        batch = pipeline([sample])
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        n_surface = batch["surface_anchor_position"].shape[1]
        n_volume = batch["volume_anchor_position"].shape[1]
        print(f"    Full mesh: {n_surface} surface, {n_volume} volume anchors")

        # Chunked inference (matching callback)
        preds = chunked_inference(model, batch, device)

        # Compute metrics: denormalize both pred and target, then compare
        sample_metrics = {}
        targets_cpu = {k: v.cpu() for k, v in batch.items() if k.endswith("_target")}

        for mode in EVALUATION_MODES:
            pred_key = f"{mode}_mean" if is_uq else mode
            pred = preds.get(pred_key)
            target = targets_cpu.get(f"{mode}_target")

            if pred is None or target is None:
                continue

            # Align sizes (chunking may truncate)
            min_n = min(pred.shape[1], target.shape[1])
            pred_aligned = pred[:, :min_n]
            target_aligned = target[:, :min_n]

            # Denormalize both (same as callback)
            denorm_pred = denormalize_field(pred_aligned, mode, normalizers)
            denorm_target = denormalize_field(target_aligned, mode, normalizers)

            field_short = mode.replace("surface_", "").replace("volume_", "vol_")
            sample_metrics.update(compute_field_metrics(denorm_pred, denorm_target, field_short))

        all_metrics.append(sample_metrics)
        all_preds.append({k: v.cpu() for k, v in preds.items()})
        all_targets.append(targets_cpu)

        print("    --- Metrics ---")
        for k, v in sorted(sample_metrics.items()):
            print(f"      {k}: {v:.6f}")

        # Full-mesh VTP visualization: query at all VTP cell centers
        # Use a training-size batch (16384 anchors) to keep memory manageable
        train_batch = train_pipeline([sample])
        train_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in train_batch.items()}

        mesh = load_surface_mesh(test_run_ids[i])
        if mesh is not None:
            print(f"    Full-mesh query inference ({mesh.n_cells} cells)...")
            cell_centers = torch.tensor(mesh.cell_centers().points, dtype=torch.float32)
            if pos_normalizer is not None:
                cell_centers_norm = pos_normalizer(cell_centers)
            else:
                cell_centers_norm = cell_centers

            # Build forward batch from training pipeline (geometry + 16k anchors)
            fwd_batch = {k: v for k, v in train_batch.items() if k in FORWARD_PROPERTIES}

            # Query at VTP cell centers in chunks
            vtp_preds = chunked_query_inference(
                model,
                fwd_batch,
                cell_centers_norm,
                device,
                query_type="surface",
            )

            # Attach denormalized predictions + GT + error to mesh
            for field, gt_key in SURFACE_GT_MAP.items():
                if gt_key not in mesh.cell_data:
                    continue
                gt = mesh.cell_data[gt_key]
                field_short = field.replace("surface_", "")

                pred_key = f"{field}_mean" if is_uq else field
                pred_norm = vtp_preds.get(pred_key)
                if pred_norm is None:
                    continue

                denorm_pred = denormalize_field(pred_norm, field, normalizers)[0].numpy().squeeze()

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
                        std_denorm = denormalize_std(torch.exp(0.5 * logvar), field, normalizers)
                        std_np = std_denorm[0].numpy()
                        if std_np.ndim > 1 and std_np.shape[-1] > 1:
                            std_np = np.linalg.norm(std_np, axis=-1)
                        mesh.cell_data[f"aleatoric_std_{field_short}"] = std_np.squeeze()

            if save_vtp:
                mesh.save(str(run_output / f"sample_{i}.vtp"))
            render_surface_plots(mesh, run_output, i, is_uq)

    with open(run_output / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n  Results saved to {run_output}")

    extra_data = {
        "is_uq": is_uq,
        "run_name": run_dir.name,
        "normalizers": normalizers,
        "preds_list": all_preds,
        "targets_list": all_targets,
    }
    return all_metrics, extra_data


def load_surface_mesh(run_id: int) -> pv.PolyData | None:
    vtp_path = SURFACE_VTP_ROOT / f"run_{run_id}" / f"boundary_{run_id}.vtp"
    if not vtp_path.exists():
        print(f"    VTP not found: {vtp_path}")
        return None
    return pv.read(str(vtp_path))


def main():
    parser = argparse.ArgumentParser(description="Post-processing for AB-UPT models")
    parser.add_argument("--run-dir", type=str, nargs="+", required=True)
    parser.add_argument("--labels", type=str, nargs="*", default=None)
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="outputs/uq_analysis")
    parser.add_argument("--save-vtp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = args.labels or [Path(r).name for r in args.run_dir]
    all_run_metrics = {}
    all_run_data = {}

    for run_path, label in zip(args.run_dir, labels):
        metrics, extra = evaluate_run(
            run_dir=Path(run_path),
            checkpoint=args.checkpoint,
            num_samples=args.num_samples,
            device=args.device,
            output_dir=output_dir,
            save_vtp=args.save_vtp,
        )
        all_run_metrics[label] = metrics
        all_run_data[label] = extra

    if len(all_run_metrics) > 1:
        print("\nGenerating comparison plots...")
        plot_comparison(all_run_metrics, output_dir)

    plot_uq_validity(all_run_data, output_dir)

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
