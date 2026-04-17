#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Epistemic uncertainty via last-layer Laplace approximation.

Steps:
1. Load trained model + training data
2. Extract last-layer features (output of decoder blocks, before LinearProjection)
3. Compute empirical Fisher information matrix H = sum(h_i @ h_i.T) over training data
4. At inference, epistemic variance = h_new.T @ H^{-1} @ h_new per query point

Works with both baseline and UQ models.

Usage:
    uv run python recipes/uncertainty_quantification/scripts/epistemic_laplace.py \
        --run-dir outputs/<run_id> \
        --label baseline_laplace --num-train-samples 50 --num-test-samples 1
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


def load_model_and_datasets(run_dir: Path, checkpoint: str, device: str):
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
    model_kind = config["model"]["kind"]
    model_config_cls = resolve_config_class(model_kind, ModelBaseConfig)
    computed = set()
    for p in model_config_cls.__mro__:
        if hasattr(p, "model_computed_fields"):
            computed |= set(p.model_computed_fields.keys())
    config["model"] = {k: v for k, v in config["model"].items() if k not in computed}
    vc = config_schema_cls(**config)

    model = Factory().instantiate(vc.model)
    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt_data["state_dict"] if "state_dict" in ckpt_data else ckpt_data
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Training dataset (for building the Hessian)
    train_config = vc.datasets["train"]
    train_dataset = DatasetFactory().instantiate(train_config)
    if train_config.pipeline is not None:
        train_pipeline = Factory().create(train_config.pipeline)
    else:
        from noether.data.pipeline import MultiStagePipeline

        train_pipeline = MultiStagePipeline()
    train_dataset.pipeline = train_pipeline

    # Test dataset
    test_config = vc.datasets["test"]
    test_dataset = DatasetFactory().instantiate(test_config)
    if test_config.pipeline is not None:
        test_pipeline = Factory().create(test_config.pipeline)
    else:
        from noether.data.pipeline import MultiStagePipeline

        test_pipeline = MultiStagePipeline()
    test_dataset.pipeline = test_pipeline

    inner = test_dataset
    while hasattr(inner, "_dataset"):
        inner = inner._dataset
    normalizers = inner.normalizers if hasattr(inner, "normalizers") else {}
    pos_normalizer = normalizers.get("surface_position")

    is_uq = hasattr(model, "forward_with_epistemic")
    return model, train_dataset, train_pipeline, test_dataset, test_pipeline, config, is_uq, normalizers, pos_normalizer


def get_backbone(model):
    """Get the inner AnchoredBranchedUPT backbone."""
    if hasattr(model, "ab_upt"):
        return model.ab_upt
    if hasattr(model, "backbone"):
        b = model.backbone
        if hasattr(b, "backbone"):
            return b.backbone  # UQ wrapper
        return b
    raise ValueError(f"Cannot find backbone in {type(model).__name__}")


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


def denormalize_std(std_tensor: torch.Tensor, field: str, normalizers: dict) -> torch.Tensor:
    if field not in normalizers:
        return std_tensor
    n = normalizers[field]
    zero = torch.zeros_like(std_tensor)
    return (n.inverse(std_tensor.cpu()) - n.inverse(zero)).abs()


def denormalize_field(tensor: torch.Tensor, field: str, normalizers: dict) -> torch.Tensor:
    if field not in normalizers:
        return tensor
    return normalizers[field].inverse(tensor.cpu())


# ---------------------------------------------------------------------------
# Feature extraction: hook into decoder blocks to capture last-layer features
# ---------------------------------------------------------------------------


class FeatureExtractor:
    """Hooks into the AB-UPT backbone to capture features BEFORE the linear decoder."""

    def __init__(self, backbone):
        self.backbone = backbone
        self.surface_features = None
        self.volume_features = None
        self._hooks = []

    def install(self):
        """Install forward hooks on the last surface and volume decoder blocks."""

        def surface_hook(module, input, output):
            # output is (x, cache) tuple from TransformerBlock
            self.surface_features = output[0].detach()

        def volume_hook(module, input, output):
            self.volume_features = output[0].detach()

        if len(self.backbone.surface_decoder_blocks) > 0:
            h = self.backbone.surface_decoder_blocks[-1].register_forward_hook(surface_hook)
            self._hooks.append(h)

        if len(self.backbone.volume_decoder_blocks) > 0:
            h = self.backbone.volume_decoder_blocks[-1].register_forward_hook(volume_hook)
            self._hooks.append(h)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Build Fisher information matrix from training data
# ---------------------------------------------------------------------------


def build_fisher(
    model,
    backbone,
    dataset,
    pipeline,
    device: str,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Compute empirical Fisher H = sum(h @ h.T) over training data.

    Returns H_surface, H_volume (both shape [hidden_dim, hidden_dim]),
    and counts of features accumulated.
    """
    hidden_dim = backbone.surface_decoder.project.in_features
    H_surface = torch.zeros(hidden_dim, hidden_dim, dtype=torch.float64, device="cpu")
    H_volume = torch.zeros(hidden_dim, hidden_dim, dtype=torch.float64, device="cpu")
    n_surface = 0
    n_volume = 0

    extractor = FeatureExtractor(backbone)
    extractor.install()

    n = min(num_samples, len(dataset))
    for i in range(n):
        if (i + 1) % 10 == 0 or i == n - 1:
            print(f"    Fisher: sample {i + 1}/{n}", end="\r")

        sample = dataset[i]
        batch = pipeline([sample])
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        fwd_batch = {k: v for k, v in batch.items() if k in FORWARD_PROPERTIES}

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device != "cpu"):
            model(**fwd_batch)

        # Accumulate outer products for surface features
        if extractor.surface_features is not None:
            h = extractor.surface_features.float().reshape(-1, hidden_dim).cpu().double()  # (N_pts, D)
            H_surface += h.T @ h  # (D, D)
            n_surface += h.shape[0]

        # Accumulate for volume features
        if extractor.volume_features is not None:
            h = extractor.volume_features.float().reshape(-1, hidden_dim).cpu().double()
            H_volume += h.T @ h
            n_volume += h.shape[0]

    print()
    extractor.remove()

    # Normalize
    if n_surface > 0:
        H_surface /= n_surface
    if n_volume > 0:
        H_volume /= n_volume

    print(f"    Fisher built: {n_surface} surface features, {n_volume} volume features")
    return H_surface, H_volume, n_surface, n_volume


# ---------------------------------------------------------------------------
# Compute Laplace epistemic variance at query points
# ---------------------------------------------------------------------------


def laplace_query_inference(
    model,
    backbone,
    fwd_batch: dict,
    query_positions: torch.Tensor,
    H_inv_surface: torch.Tensor,
    device: str,
    normalizers: dict,
    is_uq: bool,
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, np.ndarray]:
    """Compute predictions + Laplace epistemic variance at query positions.

    For each query point with feature h:
        epistemic_var = h.T @ H_inv @ h  (scalar per output dimension)
    """
    extractor = FeatureExtractor(backbone)
    extractor.install()

    n = query_positions.shape[0]
    n_chunks = max(1, (n + chunk_size - 1) // chunk_size)

    all_preds: dict[str, list] = defaultdict(list)
    all_epi_var: list[torch.Tensor] = []

    for i in range(n_chunks):
        start, end = i * chunk_size, min((i + 1) * chunk_size, n)
        chunk_batch = dict(fwd_batch)
        chunk_batch["query_surface_position"] = query_positions[start:end].unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device != "cpu"):
            out = model(**chunk_batch)

        # Collect predictions
        for key, val in out.items():
            if key.startswith("query_surface_"):
                clean_key = key.replace("query_surface_", "surface_")
                all_preds[clean_key].append(val.cpu().float())

        # Compute Laplace variance from captured features
        # The decoder block output contains [anchors, queries] concatenated.
        # We only want the query portion (last `end - start` elements).
        if extractor.surface_features is not None:
            n_query = end - start
            h_full = extractor.surface_features.float().squeeze(0).cpu()
            h = h_full[-n_query:]  # take only query features, not anchor features
            # epistemic_var per point = diag(h @ H_inv @ h.T) = sum(h * (h @ H_inv), dim=-1)
            h_proj = h.double() @ H_inv_surface  # (N_query, D)
            epi_var = (h.double() * h_proj).sum(dim=-1).float()  # (N_query,)
            all_epi_var.append(epi_var)

        if (i + 1) % 100 == 0 or i == n_chunks - 1:
            print(f"      Chunk {i + 1}/{n_chunks}", end="\r")

    print()
    extractor.remove()

    preds = {k: torch.cat(v, dim=1) for k, v in all_preds.items()}
    epi_var = torch.cat(all_epi_var, dim=0)  # (N_total,) — scalar variance per point

    # Convert to per-field epistemic std (scale by decoder weight norms)
    result = {}
    for field, gt_key in SURFACE_GT_MAP.items():
        pred_key = f"{field}_mean" if is_uq else field
        if pred_key in preds:
            denorm_pred = denormalize_field(preds[pred_key], field, normalizers)[0].numpy().squeeze()
            result[f"pred_{field}"] = denorm_pred

            # Epistemic std in output space:
            # Var[y_j] = W_j @ H_inv @ W_j.T per output dim j, per point
            # We compute this properly: for each point with feature h,
            # the full output variance per output dim is: w_j.T @ H_inv @ (h h.T) @ H_inv @ w_j
            # But we already have epi_var = h.T @ H_inv @ h (scalar per point)
            # So Var[y_j] ≈ ||w_j||^2 * epi_var (approximation assuming H_inv ≈ scaled identity)
            W = backbone.surface_decoder.project.weight.detach().cpu().float()

            field_slices = backbone.data_specs.surface_output_dims.field_slices
            field_name = field.replace("surface_", "")
            if field_name in field_slices:
                slc = field_slices[field_name]
                W_field = W[slc]  # (D_field, hidden_dim)
                w_norm_sq = (W_field**2).sum().item()
            else:
                w_norm_sq = (W**2).sum().item()

            # epi_var is in normalized feature space, W maps to normalized output space
            # To get physical units: multiply by the field's denormalization scale^2
            # denormalize_std for a scalar: use inverse(1) - inverse(0) to get the scale
            one = torch.ones(1, 1, 1)
            zero = torch.zeros(1, 1, 1)
            if field in normalizers:
                scale = (normalizers[field].inverse(one) - normalizers[field].inverse(zero)).abs().mean().item()
            else:
                scale = 1.0

            epi_std_physical = (torch.sqrt(epi_var * w_norm_sq) * scale).numpy()
            result[f"epistemic_std_{field}"] = epi_std_physical

        if is_uq and f"{field}_log_var" in preds:
            log_var = preds[f"{field}_log_var"]
            std_norm = torch.exp(0.5 * log_var)
            # Denormalize std: scale by field's denorm scale, no shift
            if field in normalizers:
                one = torch.ones_like(std_norm)
                zero = torch.zeros_like(std_norm)
                field_scale = (normalizers[field].inverse(one.cpu()) - normalizers[field].inverse(zero.cpu())).abs()
                ale_std = (std_norm.cpu() * field_scale)[0].numpy()
            else:
                ale_std = std_norm[0].cpu().numpy()
            if ale_std.ndim > 1 and ale_std.shape[-1] > 1:
                ale_std = np.linalg.norm(ale_std, axis=-1)
            result[f"aleatoric_std_{field}"] = ale_std.squeeze()

    return result


# ---------------------------------------------------------------------------
# Visualization
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
        epi_key = f"laplace_std_{field_short}"
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
                    "Laplace \u03c3",
                    epi_key,
                    "Reds",
                    [float(np.percentile(epi_data, 5)), float(np.percentile(epi_data, 95))],
                )
            )
        if is_uq and ale_key in mesh.cell_data:
            ale_data = mesh.cell_data[ale_key]
            panels.append(
                (
                    "Aleatoric \u03c3",
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
        img_path = output_path / f"laplace_{field_short}_sample{sample_idx}.png"
        fig.savefig(img_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Rendered: {img_path.name}")


def plot_scatter(mesh: pv.PolyData, output_path: Path, sample_idx: int, is_uq: bool):
    for field, _ in SURFACE_GT_MAP.items():
        field_short = field.replace("surface_", "")
        error_key = f"error_{field_short}"
        epi_key = f"laplace_std_{field_short}"
        ale_key = f"aleatoric_std_{field_short}"

        if error_key not in mesh.cell_data or epi_key not in mesh.cell_data:
            continue

        error = mesh.cell_data[error_key]
        epi_std = mesh.cell_data[epi_key]

        n_cols = 2 if (is_uq and ale_key in mesh.cell_data) else 1
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
        if n_cols == 1:
            axes = [axes]

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
        ax.set_xlabel("Laplace \u03c3 (binned)")
        ax.set_ylabel("Mean |error|")
        ax.set_title(f"Laplace Epistemic vs Error \u2014 {field_short}")
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
            ax.set_xlabel("Aleatoric \u03c3 (binned)")
            ax.set_ylabel("Mean |error|")
            ax.set_title(f"Aleatoric vs Error \u2014 {field_short}")
            ax.legend()

        plt.tight_layout()
        fig.savefig(output_path / f"laplace_scatter_{field_short}_sample{sample_idx}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: laplace_scatter_{field_short}_sample{sample_idx}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Epistemic UQ via last-layer Laplace approximation")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--num-train-samples", type=int, default=50, help="Training samples for Fisher computation")
    parser.add_argument("--num-test-samples", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="outputs/epistemic_laplace")
    parser.add_argument("--save-vtp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    label = args.label or run_dir.name
    output_dir = Path(args.output_dir) / label
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"Last-Layer Laplace Epistemic UQ: {label}")
    print(f"{'=' * 60}")

    model, train_ds, train_pipeline, test_ds, test_pipeline, config, is_uq, normalizers, pos_normalizer = (
        load_model_and_datasets(run_dir, args.checkpoint, args.device)
    )
    backbone = get_backbone(model)
    hidden_dim = backbone.surface_decoder.project.in_features
    print(f"  Model: {'UQ' if is_uq else 'Baseline'}, hidden_dim={hidden_dim}")

    # Step 1: Build Fisher from training data
    print(f"\n  Step 1: Building Fisher matrix from {args.num_train_samples} training samples...")
    H_surface, H_volume, n_surf, n_vol = build_fisher(
        model, backbone, train_ds, train_pipeline, args.device, args.num_train_samples
    )

    # Add small regularization for numerical stability
    reg = 1e-3
    H_surface += reg * torch.eye(hidden_dim, dtype=torch.float64)
    H_inv_surface = torch.linalg.inv(H_surface)
    print(f"    H_surface condition number: {torch.linalg.cond(H_surface):.1f}")
    print(f"    H_inv_surface range: [{H_inv_surface.min():.6f}, {H_inv_surface.max():.6f}]")

    # Save Fisher
    torch.save({"H_surface": H_surface, "H_inv_surface": H_inv_surface}, output_dir / "fisher.pt")

    # Step 2: Inference on test samples
    test_run_ids = get_test_run_ids(test_ds)
    n_test = min(args.num_test_samples, len(test_ds))
    all_sample_metrics = []

    for i in range(n_test):
        print(f"\n  Step 2: Test sample {i + 1}/{n_test} (run_{test_run_ids[i]})")

        mesh = load_surface_mesh(test_run_ids[i])
        if mesh is None:
            continue

        cell_centers = torch.tensor(mesh.cell_centers().points, dtype=torch.float32)
        if pos_normalizer is not None:
            query_positions = pos_normalizer(cell_centers)
        else:
            query_positions = cell_centers

        # Build batch from training pipeline (geometry + anchors)
        sample = test_ds[i]
        batch = test_pipeline([sample])
        batch = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        fwd_batch = {k: v for k, v in batch.items() if k in FORWARD_PROPERTIES}

        print(f"    Laplace inference at {mesh.n_cells} query points...")
        results = laplace_query_inference(
            model,
            backbone,
            fwd_batch,
            query_positions,
            H_inv_surface,
            args.device,
            normalizers,
            is_uq,
        )

        # Attach to mesh
        sample_metrics = {}
        for field, gt_key in SURFACE_GT_MAP.items():
            if gt_key not in mesh.cell_data:
                continue
            gt = mesh.cell_data[gt_key]
            field_short = field.replace("surface_", "")

            pred = results.get(f"pred_{field}")
            epi_std = results.get(f"epistemic_std_{field}")
            if pred is None:
                continue

            if pred.ndim > 1 and pred.shape[-1] > 1:
                mesh.cell_data[f"pred_{field_short}_mag"] = np.linalg.norm(pred, axis=-1)
                mesh.cell_data[f"gt_{field_short}_mag"] = np.linalg.norm(gt, axis=-1)
                mesh.cell_data[f"error_{field_short}"] = np.linalg.norm(pred - gt, axis=-1)
            else:
                mesh.cell_data[f"pred_{field_short}"] = pred.squeeze()
                mesh.cell_data[f"error_{field_short}"] = np.abs(pred.squeeze() - gt.squeeze())

            if epi_std is not None:
                if epi_std.ndim > 1 and epi_std.shape[-1] > 1:
                    mesh.cell_data[f"laplace_std_{field_short}"] = np.linalg.norm(epi_std, axis=-1)
                else:
                    mesh.cell_data[f"laplace_std_{field_short}"] = epi_std.squeeze()

            ale_std = results.get(f"aleatoric_std_{field}")
            if ale_std is not None:
                if ale_std.ndim > 1 and ale_std.shape[-1] > 1:
                    mesh.cell_data[f"aleatoric_std_{field_short}"] = np.linalg.norm(ale_std, axis=-1)
                else:
                    mesh.cell_data[f"aleatoric_std_{field_short}"] = ale_std.squeeze()

            # Metrics
            err = mesh.cell_data[f"error_{field_short}"]
            epi = mesh.cell_data.get(f"laplace_std_{field_short}")
            if epi is not None:
                sample_metrics[f"{field_short}_error_mean"] = float(err.mean())
                sample_metrics[f"{field_short}_laplace_std_mean"] = float(epi.mean())
                sample_metrics[f"{field_short}_laplace_std_median"] = float(np.median(epi))
                sample_metrics[f"{field_short}_error_epi_correlation"] = float(
                    np.corrcoef(err.ravel(), epi.ravel())[0, 1]
                )

        print("    --- Metrics ---")
        for k, v in sorted(sample_metrics.items()):
            print(f"      {k}: {v:.6f}")

        render_maps(mesh, output_dir, i, is_uq)
        plot_scatter(mesh, output_dir, i, is_uq)

        if args.save_vtp:
            mesh.save(str(output_dir / f"sample_{i}.vtp"))

        all_sample_metrics.append(sample_metrics)

        with open(output_dir / "metrics.json", "w") as f:
            json.dump(all_sample_metrics, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
