#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Epistemic uncertainty via geometry supernode resampling.

For each of K runs, re-runs the pipeline to get a different random set of
geometry points and supernodes, then queries at fixed VTP cell centers.
The variance across K runs = epistemic uncertainty from geometry encoding.

No retraining needed — uses the same trained checkpoint.

Usage:
    uv run python recipes/uncertainty_quantification/scripts/epistemic_geometry.py \
        --run-dir outputs/<run_id> \
        --label baseline_geo --num-samples 1 --num-subsamples 10
"""

from __future__ import annotations

import argparse
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
FORWARD_PROPERTIES = [
    "geometry_position",
    "geometry_supernode_idx",
    "geometry_batch_idx",
    "surface_anchor_position",
    "volume_anchor_position",
]
SURFACE_GT_MAP = {
    "surface_pressure": "pMeanTrim",
    "surface_friction": "wallShearStressMeanTrim",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_model_and_data(run_dir: Path, checkpoint: str, device: str):
    stage_dir = run_dir / "train"
    with open(stage_dir / "hp_resolved.yaml") as f:
        config = yaml.full_load(f)

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

    model = Factory().instantiate(validated_config.model)
    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt_data["state_dict"] if "state_dict" in ckpt_data else ckpt_data
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    test_config = validated_config.datasets["test"]
    dataset = DatasetFactory().instantiate(test_config)
    if test_config.pipeline is not None:
        pipeline = Factory().create(test_config.pipeline)
    else:
        from noether.data.pipeline import MultiStagePipeline

        pipeline = MultiStagePipeline()
    dataset.pipeline = pipeline

    inner = dataset
    while hasattr(inner, "_dataset"):
        inner = inner._dataset
    normalizers = inner.normalizers if hasattr(inner, "normalizers") else {}
    pos_normalizer = normalizers.get("surface_position")

    is_uq = hasattr(model, "forward_with_epistemic")
    return model, dataset, pipeline, config, is_uq, normalizers, pos_normalizer


def get_test_run_ids(dataset) -> list[int]:
    inner = dataset
    while hasattr(inner, "_dataset"):
        inner = inner._dataset
    return sorted(inner.get_dataset_splits.test)


def load_surface_mesh(run_id: int) -> pv.PolyData | None:
    vtp_path = SURFACE_VTP_ROOT / f"run_{run_id}" / f"boundary_{run_id}.vtp"
    if not vtp_path.exists():
        return None
    return pv.read(str(vtp_path))


def denormalize_field(tensor: torch.Tensor, field: str, normalizers: dict) -> torch.Tensor:
    if field not in normalizers:
        return tensor
    return normalizers[field].inverse(tensor.cpu())


def denormalize_std(std_tensor: torch.Tensor, field: str, normalizers: dict) -> torch.Tensor:
    if field not in normalizers:
        return std_tensor
    n = normalizers[field]
    zero = torch.zeros_like(std_tensor)
    return (n.inverse(std_tensor.cpu()) - n.inverse(zero)).abs()


# ---------------------------------------------------------------------------
# Query inference (same as uq_postprocessing)
# ---------------------------------------------------------------------------


def chunked_query_inference(
    model,
    fwd_batch: dict,
    query_positions: torch.Tensor,
    device: str,
    query_type: str = "surface",
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, torch.Tensor]:
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
                clean_key = key.replace(prefix, f"{query_type}_")
                outputs[clean_key].append(val.cpu().float())

        if (i + 1) % 100 == 0 or i == n_chunks - 1:
            print(f"      Chunk {i + 1}/{n_chunks}", end="\r")

    if n_chunks > 1:
        print()
    return {key: torch.cat(chunks, dim=1) for key, chunks in outputs.items()}


# ---------------------------------------------------------------------------
# Epistemic inference: K pipeline runs with different geometry resampling
# ---------------------------------------------------------------------------


def epistemic_geometry_inference(
    model,
    dataset,
    pipeline,
    sample_idx: int,
    query_positions: torch.Tensor,
    device: str,
    num_subsamples: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Run K forward passes, each with a freshly resampled pipeline batch.

    Each call to pipeline([sample]) produces:
    - Different random geometry points (from the full surface mesh)
    - Different random supernodes (from the geometry points)
    - Different random surface/volume anchors

    All runs query at the same fixed VTP cell centers.
    """
    sample = dataset[sample_idx]
    all_preds: list[dict[str, torch.Tensor]] = []

    for k in range(num_subsamples):
        print(f"    Run {k + 1}/{num_subsamples}: resampling geometry...")

        # Each pipeline call randomizes geometry/supernodes/anchors
        batch = pipeline([sample])
        batch = {key: v.to(device) if torch.is_tensor(v) else v for key, v in batch.items()}

        n_geo = batch["geometry_position"].shape[0]
        n_sn = batch["geometry_supernode_idx"].shape[0]
        n_surf = batch["surface_anchor_position"].shape[1]
        n_vol = batch["volume_anchor_position"].shape[1]
        if k == 0:
            print(f"      Geometry: {n_geo} pts, {n_sn} supernodes, {n_surf} surf anchors, {n_vol} vol anchors")

        fwd_batch = {key: v for key, v in batch.items() if key in FORWARD_PROPERTIES}
        preds = chunked_query_inference(model, fwd_batch, query_positions, device)
        all_preds.append(preds)

    # Compute mean and variance across K runs
    keys = all_preds[0].keys()
    mean_preds = {}
    var_preds = {}
    for key in keys:
        stacked = torch.stack([p[key] for p in all_preds], dim=0)  # (K, 1, N, D)
        mean_preds[key] = stacked.mean(dim=0)
        var_preds[key] = stacked.var(dim=0)

    return mean_preds, var_preds, all_preds


# ---------------------------------------------------------------------------
# Visualization (same as epistemic_investigation.py)
# ---------------------------------------------------------------------------


def render_maps(mesh: pv.PolyData, output_path: Path, sample_idx: int, is_uq: bool):
    pv.OFF_SCREEN = True

    for field, gt_key in SURFACE_GT_MAP.items():
        field_short = field.replace("surface_", "")

        gt_plot_key = f"gt_{field_short}_mag" if f"gt_{field_short}_mag" in mesh.cell_data else gt_key
        pred_plot_key = (
            f"pred_{field_short}_mag" if f"pred_{field_short}_mag" in mesh.cell_data else f"pred_{field_short}"
        )
        error_key = f"error_{field_short}"
        epi_key = f"epistemic_std_{field_short}"
        ale_key = f"aleatoric_std_{field_short}"

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
        if epi_key in mesh.cell_data:
            epi_data = mesh.cell_data[epi_key]
            panels.append(
                (
                    "Epistemic σ",
                    epi_key,
                    "Reds",
                    [float(np.percentile(epi_data, 5)), float(np.percentile(epi_data, 95))],
                )
            )
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
        img_path = output_path / f"geo_epistemic_{field_short}_sample{sample_idx}.png"
        fig.savefig(img_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Rendered: {img_path.name}")


def plot_scatter(mesh: pv.PolyData, output_path: Path, sample_idx: int, is_uq: bool):
    for field, _ in SURFACE_GT_MAP.items():
        field_short = field.replace("surface_", "")
        error_key = f"error_{field_short}"
        epi_key = f"epistemic_std_{field_short}"
        ale_key = f"aleatoric_std_{field_short}"

        if error_key not in mesh.cell_data or epi_key not in mesh.cell_data:
            continue

        error = mesh.cell_data[error_key]
        epi_std = mesh.cell_data[epi_key]

        n_cols = 2 if (is_uq and ale_key in mesh.cell_data) else 1
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
        if n_cols == 1:
            axes = [axes]

        # Epistemic vs error
        ax = axes[0]
        n_bins = 30
        bins = np.percentile(epi_std, np.linspace(0, 100, n_bins + 1))
        bc, be = [], []
        for j in range(n_bins):
            mask = (epi_std >= bins[j]) & (epi_std < bins[j + 1])
            if mask.sum() > 100:
                bc.append(epi_std[mask].mean())
                be.append(error[mask].mean())
        ax.scatter(bc, be, s=30, edgecolors="k", linewidths=0.5)
        if bc:
            lim = max(max(bc), max(be)) * 1.1
            ax.plot([0, lim], [0, lim], "r--", lw=2, label="y = x")
        ax.set_xlabel("Epistemic σ (binned)")
        ax.set_ylabel("Mean |error|")
        ax.set_title(f"Epistemic (geometry) vs Error — {field_short}")
        ax.legend()

        if n_cols == 2:
            ax = axes[1]
            ale_std = mesh.cell_data[ale_key]
            bins = np.percentile(ale_std, np.linspace(0, 100, n_bins + 1))
            bc, be = [], []
            for j in range(n_bins):
                mask = (ale_std >= bins[j]) & (ale_std < bins[j + 1])
                if mask.sum() > 100:
                    bc.append(ale_std[mask].mean())
                    be.append(error[mask].mean())
            ax.scatter(bc, be, s=30, edgecolors="k", linewidths=0.5, color="orange")
            if bc:
                lim = max(max(bc), max(be)) * 1.1
                ax.plot([0, lim], [0, lim], "r--", lw=2, label="y = x")
            ax.set_xlabel("Aleatoric σ (binned)")
            ax.set_ylabel("Mean |error|")
            ax.set_title(f"Aleatoric vs Error — {field_short}")
            ax.legend()

        plt.tight_layout()
        fig.savefig(output_path / f"geo_scatter_{field_short}_sample{sample_idx}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: geo_scatter_{field_short}_sample{sample_idx}.png")


def plot_histogram(
    all_preds: list[dict[str, torch.Tensor]],
    normalizers: dict,
    output_path: Path,
    sample_idx: int,
):
    for field, _ in SURFACE_GT_MAP.items():
        field_short = field.replace("surface_", "")
        key = f"{field}_mean" if f"{field}_mean" in all_preds[0] else field
        if key not in all_preds[0]:
            continue

        stacked = torch.stack([p[key] for p in all_preds], dim=0)
        std_norm = stacked.std(dim=0).squeeze()
        std_phys = denormalize_std(std_norm.unsqueeze(0), field, normalizers)[0].numpy()
        if std_phys.ndim > 1 and std_phys.shape[-1] > 1:
            std_phys = np.linalg.norm(std_phys, axis=-1)
        std_phys = std_phys.squeeze()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(std_phys, bins=100, density=True, alpha=0.7, edgecolor="k", linewidth=0.3)
        ax.axvline(np.median(std_phys), color="r", linestyle="--", label=f"Median = {np.median(std_phys):.4f}")
        ax.axvline(np.mean(std_phys), color="b", linestyle="--", label=f"Mean = {np.mean(std_phys):.4f}")
        ax.set_xlabel(f"Epistemic σ ({field_short}, physical units)")
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution of Geometry-Epistemic Uncertainty — {field_short}")
        ax.legend()
        plt.tight_layout()
        fig.savefig(output_path / f"geo_hist_{field_short}_sample{sample_idx}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: geo_hist_{field_short}_sample{sample_idx}.png")


def plot_epi_vs_ale(mesh: pv.PolyData, output_path: Path, sample_idx: int):
    for field, _ in SURFACE_GT_MAP.items():
        field_short = field.replace("surface_", "")
        epi_key = f"epistemic_std_{field_short}"
        ale_key = f"aleatoric_std_{field_short}"
        if epi_key not in mesh.cell_data or ale_key not in mesh.cell_data:
            continue

        epi = mesh.cell_data[epi_key]
        ale = mesh.cell_data[ale_key]

        fig, ax = plt.subplots(figsize=(6, 5))
        idx = np.random.choice(len(epi), size=min(50000, len(epi)), replace=False)
        ax.scatter(ale[idx], epi[idx], s=1, alpha=0.3)
        ax.set_xlabel("Aleatoric σ")
        ax.set_ylabel("Epistemic σ (geometry)")
        ax.set_title(f"Epistemic vs Aleatoric — {field_short}")
        corr = np.corrcoef(ale, epi)[0, 1]
        ax.text(
            0.05,
            0.95,
            f"Pearson r = {corr:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat"),
        )
        plt.tight_layout()
        fig.savefig(output_path / f"geo_epi_vs_ale_{field_short}_sample{sample_idx}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: geo_epi_vs_ale_{field_short}_sample{sample_idx}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Epistemic UQ via geometry supernode resampling")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--num-subsamples", type=int, default=10, help="K: number of geometry resamplings")
    parser.add_argument("--output-dir", type=str, default="outputs/epistemic_geometry")
    parser.add_argument("--save-vtp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    label = args.label or run_dir.name
    output_dir = Path(args.output_dir) / label
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"Epistemic UQ via Geometry Resampling: {label}")
    print(f"  K={args.num_subsamples}")
    print(f"{'=' * 60}")

    model, dataset, pipeline, config, is_uq, normalizers, pos_normalizer = load_model_and_data(
        run_dir, args.checkpoint, args.device
    )
    print(f"  Model: {'UQ' if is_uq else 'Baseline'}")

    test_run_ids = get_test_run_ids(dataset)
    n = min(args.num_samples, len(dataset))

    for i in range(n):
        print(f"\n  Sample {i + 1}/{n} (run_{test_run_ids[i]})")

        mesh = load_surface_mesh(test_run_ids[i])
        if mesh is None:
            print("    Skipping: no VTP mesh")
            continue

        cell_centers = torch.tensor(mesh.cell_centers().points, dtype=torch.float32)
        if pos_normalizer is not None:
            query_positions = pos_normalizer(cell_centers)
        else:
            query_positions = cell_centers

        print(f"    Running {args.num_subsamples} geometry-resampled inferences at {mesh.n_cells} query points...")
        mean_preds, var_preds, all_preds = epistemic_geometry_inference(
            model,
            dataset,
            pipeline,
            i,
            query_positions,
            args.device,
            args.num_subsamples,
        )

        # Attach to mesh
        sample_metrics = {}
        for field, gt_key in SURFACE_GT_MAP.items():
            if gt_key not in mesh.cell_data:
                continue
            gt = mesh.cell_data[gt_key]
            field_short = field.replace("surface_", "")

            pred_key = f"{field}_mean" if is_uq else field
            if pred_key not in mean_preds:
                continue

            denorm_pred = denormalize_field(mean_preds[pred_key], field, normalizers)[0].numpy().squeeze()
            epi_std = denormalize_std(torch.sqrt(var_preds[pred_key]), field, normalizers)[0].numpy().squeeze()

            if denorm_pred.ndim > 1 and denorm_pred.shape[-1] > 1:
                mesh.cell_data[f"pred_{field_short}_mag"] = np.linalg.norm(denorm_pred, axis=-1)
                mesh.cell_data[f"gt_{field_short}_mag"] = np.linalg.norm(gt, axis=-1)
                mesh.cell_data[f"error_{field_short}"] = np.linalg.norm(denorm_pred - gt, axis=-1)
                mesh.cell_data[f"epistemic_std_{field_short}"] = np.linalg.norm(epi_std, axis=-1)
            else:
                mesh.cell_data[f"pred_{field_short}"] = denorm_pred
                mesh.cell_data[f"error_{field_short}"] = np.abs(denorm_pred - gt.squeeze())
                mesh.cell_data[f"epistemic_std_{field_short}"] = epi_std

            if is_uq and f"{field}_log_var" in mean_preds:
                ale_std = (
                    denormalize_std(torch.exp(0.5 * mean_preds[f"{field}_log_var"]), field, normalizers)[0]
                    .numpy()
                    .squeeze()
                )
                if ale_std.ndim > 1 and ale_std.shape[-1] > 1:
                    mesh.cell_data[f"aleatoric_std_{field_short}"] = np.linalg.norm(ale_std, axis=-1)
                else:
                    mesh.cell_data[f"aleatoric_std_{field_short}"] = ale_std

            err = mesh.cell_data[f"error_{field_short}"]
            epi = mesh.cell_data[f"epistemic_std_{field_short}"]
            sample_metrics[f"{field_short}_error_mean"] = float(err.mean())
            sample_metrics[f"{field_short}_epistemic_std_mean"] = float(epi.mean())
            sample_metrics[f"{field_short}_epistemic_std_median"] = float(np.median(epi))
            sample_metrics[f"{field_short}_error_epi_correlation"] = float(np.corrcoef(err, epi)[0, 1])
            if f"aleatoric_std_{field_short}" in mesh.cell_data:
                ale = mesh.cell_data[f"aleatoric_std_{field_short}"]
                sample_metrics[f"{field_short}_aleatoric_std_mean"] = float(ale.mean())
                sample_metrics[f"{field_short}_epi_ale_correlation"] = float(np.corrcoef(epi, ale)[0, 1])

        print("    --- Metrics ---")
        for k, v in sorted(sample_metrics.items()):
            print(f"      {k}: {v:.6f}")

        render_maps(mesh, output_dir, i, is_uq)
        plot_scatter(mesh, output_dir, i, is_uq)
        plot_histogram(all_preds, normalizers, output_dir, i)
        if is_uq:
            plot_epi_vs_ale(mesh, output_dir, i)

        if args.save_vtp:
            mesh.save(str(output_dir / f"sample_{i}.vtp"))

        with open(output_dir / "metrics.json", "w") as f:
            json.dump(sample_metrics, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
