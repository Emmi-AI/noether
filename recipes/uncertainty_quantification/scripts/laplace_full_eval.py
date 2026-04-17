#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Full Laplace + Aleatoric evaluation across test + val splits.

Runs both shared and per-field Laplace + aleatoric (if UQ model) on test and val.
Produces per-sample metrics and combined scatter plots with test/val colors.

Usage:
    uv run python recipes/uncertainty_quantification/scripts/laplace_full_eval.py \
        --run-dir outputs/<run_id> --label baseline_full
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
SPLIT_COLORS = {"test": "C0", "val": "C1"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_all(run_dir, checkpoint, device):
    stage_dir = run_dir / "train"
    with open(stage_dir / "hp_resolved.yaml") as f:
        config = yaml.full_load(f)

    ckpt_dir = stage_dir / "checkpoints"
    if checkpoint == "best":
        files = list(ckpt_dir.glob("*best*model.th"))
        ckpt_path = files[0] if files else next(ckpt_dir.glob("*latest_model.th"))
    elif checkpoint == "latest":
        ckpt_path = next(ckpt_dir.glob("*latest_model.th"))
    else:
        ckpt_path = ckpt_dir / checkpoint
    print(f"  Checkpoint: {ckpt_path.name}")

    schema_cls = class_constructor_from_class_path(
        config.get("config_schema_kind", "noether.core.schemas.schema.ConfigSchema")
    )
    model_cls = resolve_config_class(config["model"]["kind"], ModelBaseConfig)
    computed = set()
    for p in model_cls.__mro__:
        if hasattr(p, "model_computed_fields"):
            computed |= set(p.model_computed_fields.keys())
    config["model"] = {k: v for k, v in config["model"].items() if k not in computed}
    vc = schema_cls(**config)

    model = Factory().instantiate(vc.model)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
    model.to(device).eval()

    datasets, pipelines = {}, {}
    for split in ["train", "test", "val"]:
        if split not in vc.datasets:
            continue
        ds = DatasetFactory().instantiate(vc.datasets[split])
        pl = Factory().create(vc.datasets[split].pipeline) if vc.datasets[split].pipeline else None
        if pl:
            ds.pipeline = pl
        datasets[split] = ds
        pipelines[split] = pl

    inner = datasets.get("test", datasets.get("val"))
    while hasattr(inner, "_dataset"):
        inner = inner._dataset
    normalizers = inner.normalizers if hasattr(inner, "normalizers") else {}

    is_uq = hasattr(model, "forward_with_epistemic")
    return model, datasets, pipelines, config, is_uq, normalizers, normalizers.get("surface_position")


def get_backbone(model):
    if hasattr(model, "ab_upt"):
        return model.ab_upt
    b = getattr(model, "backbone", None)
    if b and hasattr(b, "backbone"):
        return b.backbone
    if b:
        return b
    raise ValueError(f"No backbone in {type(model).__name__}")


def get_run_ids(dataset):
    inner = dataset
    while hasattr(inner, "_dataset"):
        inner = inner._dataset
    splits = inner.get_dataset_splits
    return {
        "test": sorted(splits.test) if hasattr(splits, "test") else [],
        "val": sorted(splits.val) if hasattr(splits, "val") else [],
    }


def load_mesh(rid):
    p = SURFACE_VTP_ROOT / f"run_{rid}" / f"boundary_{rid}.vtp"
    return pv.read(str(p)) if p.exists() else None


def denorm(t, field, norms):
    return norms[field].inverse(t.cpu()) if field in norms else t


def denorm_scale(field, norms):
    """Get denormalization scale factor (no shift) for a field."""
    if field not in norms:
        return 1.0
    one, zero = torch.ones(1, 1, 1), torch.zeros(1, 1, 1)
    return (norms[field].inverse(one) - norms[field].inverse(zero)).abs().mean().item()


# ---------------------------------------------------------------------------
# Feature hook
# ---------------------------------------------------------------------------


class FeatureHook:
    def __init__(self, backbone):
        self.features = None
        self._h = []
        if len(backbone.surface_decoder_blocks) > 0:
            self._h.append(
                backbone.surface_decoder_blocks[-1].register_forward_hook(
                    lambda m, inp, out: setattr(self, "features", out[0].detach())
                )
            )

    def remove(self):
        for h in self._h:
            h.remove()


# ---------------------------------------------------------------------------
# Fisher
# ---------------------------------------------------------------------------


def build_fisher(model, backbone, ds, pl, device, n_samples, reg=1e-3):
    D = backbone.surface_decoder.project.in_features
    H = torch.zeros(D, D, dtype=torch.float64)
    cnt = 0
    hook = FeatureHook(backbone)
    n = min(n_samples, len(ds))
    for i in range(n):
        if (i + 1) % 10 == 0 or i == n - 1:
            print(f"    Fisher: {i + 1}/{n}", end="\r")
        batch = pl([ds[i]])
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device != "cpu"):
            model(**{k: v for k, v in batch.items() if k in FORWARD_PROPERTIES})
        if hook.features is not None:
            h = hook.features.float().reshape(-1, D).cpu().double()
            H += h.T @ h
            cnt += h.shape[0]
    print()
    hook.remove()
    if cnt > 0:
        H /= cnt
    H += reg * torch.eye(D, dtype=torch.float64)
    print(f"    Fisher: {cnt} features, cond={torch.linalg.cond(H):.1f}")
    return torch.linalg.inv(H)


# ---------------------------------------------------------------------------
# Inference: shared + per-field Laplace + aleatoric in one pass
# ---------------------------------------------------------------------------


def laplace_inference(model, backbone, fwd_batch, qpos, H_inv, device, norms, is_uq, chunk_size=CHUNK_SIZE):
    hook = FeatureHook(backbone)
    W = backbone.surface_decoder.project.weight.detach().cpu().double()
    slices = backbone.data_specs.surface_output_dims.field_slices
    H_inv_d = H_inv.double()

    WH_field = {}
    for field in SURFACE_GT_MAP:
        fn = field.replace("surface_", "")
        s = slices.get(fn)
        WH_field[field] = (W[s] if s else W) @ H_inv_d

    w_norm_sq = float((W**2).sum())

    n = qpos.shape[0]
    n_ch = max(1, (n + chunk_size - 1) // chunk_size)
    preds: dict[str, list] = defaultdict(list)
    shared_epi: list = []
    pf_epi: dict[str, list] = defaultdict(list)

    for i in range(n_ch):
        s, e = i * chunk_size, min((i + 1) * chunk_size, n)
        nq = e - s
        cb = dict(fwd_batch)
        cb["query_surface_position"] = qpos[s:e].unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device != "cpu"):
            out = model(**cb)

        for k, v in out.items():
            if k.startswith("query_surface_"):
                preds[k.replace("query_surface_", "surface_")].append(v.cpu().float())

        if hook.features is not None:
            h = hook.features.float().squeeze(0)[-nq:].cpu().double()
            hp = h @ H_inv_d
            shared_epi.append(torch.sqrt((h * hp).sum(dim=-1).float() * w_norm_sq).clamp(min=0))
            for field, wh in WH_field.items():
                proj = h @ wh.T
                pf_epi[field].append(torch.sqrt((proj**2).sum(dim=-1).float().clamp(min=0)))

        if (i + 1) % 100 == 0 or i == n_ch - 1:
            print(f"      Chunk {i + 1}/{n_ch}", end="\r")

    print()
    hook.remove()

    preds = {k: torch.cat(v, dim=1) for k, v in preds.items()}
    shared_std = torch.cat(shared_epi, dim=0)
    pf_stds = {k: torch.cat(v, dim=0) for k, v in pf_epi.items()}

    result = {}
    for field in SURFACE_GT_MAP:
        pk = f"{field}_mean" if is_uq else field
        sc = denorm_scale(field, norms)
        if pk in preds:
            result[f"pred_{field}"] = denorm(preds[pk], field, norms)[0].numpy().squeeze()
        result[f"shared_std_{field}"] = (shared_std * sc).numpy()
        if field in pf_stds:
            result[f"pf_std_{field}"] = (pf_stds[field] * sc).numpy()

        # Aleatoric
        if is_uq and f"{field}_log_var" in preds:
            sn = torch.exp(0.5 * preds[f"{field}_log_var"])
            if field in norms:
                o, z = torch.ones_like(sn), torch.zeros_like(sn)
                ale = (sn.cpu() * (norms[field].inverse(o.cpu()) - norms[field].inverse(z.cpu())).abs())[0].numpy()
            else:
                ale = sn[0].cpu().numpy()
            if ale.ndim > 1 and ale.shape[-1] > 1:
                ale = np.linalg.norm(ale, axis=-1)
            result[f"ale_std_{field}"] = ale.squeeze()

    return result


# ---------------------------------------------------------------------------
# Process one sample
# ---------------------------------------------------------------------------


def process_sample(model, backbone, dataset, pipeline, run_id, idx, H_inv, device, norms, pos_norm, is_uq, split):
    mesh = load_mesh(run_id)
    if mesh is None:
        return None

    cc = torch.tensor(mesh.cell_centers().points, dtype=torch.float32)
    qp = pos_norm(cc) if pos_norm else cc

    batch = pipeline([dataset[idx]])
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    fwd = {k: v for k, v in batch.items() if k in FORWARD_PROPERTIES}

    print(f"    Inference at {mesh.n_cells} pts...")
    res = laplace_inference(model, backbone, fwd, qp, H_inv, device, norms, is_uq)

    metrics = {"split": split, "run_id": int(run_id)}
    for field, gtk in SURFACE_GT_MAP.items():
        if gtk not in mesh.cell_data:
            continue
        gt = mesh.cell_data[gtk]
        fs = field.replace("surface_", "")
        pred = res.get(f"pred_{field}")
        if pred is None:
            continue

        err = (
            np.abs(pred.squeeze() - gt.squeeze())
            if pred.ndim <= 1 or pred.shape[-1] == 1
            else np.linalg.norm(pred - gt, axis=-1)
        )

        metrics[f"{fs}_error_mean"] = float(err.mean())

        for mkey, skey in [("shared", f"shared_std_{field}"), ("pf", f"pf_std_{field}"), ("ale", f"ale_std_{field}")]:
            std = res.get(skey)
            if std is not None:
                std_flat = std.ravel()
                err_flat = err.ravel()
                n = min(len(std_flat), len(err_flat))
                metrics[f"{fs}_{mkey}_corr"] = float(np.corrcoef(err_flat[:n], std_flat[:n])[0, 1])
                metrics[f"{fs}_{mkey}_std_mean"] = float(std.mean())

    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_scatter_all(all_metrics, out_dir, is_uq):
    """For each field × method: scatter σ vs error, test=blue, val=orange."""
    methods = [("shared", "Shared Laplace"), ("pf", "Per-field Laplace")]
    if is_uq:
        methods.append(("ale", "Aleatoric"))

    for fs in ["pressure", "friction"]:
        n_methods = len(methods)
        fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
        if n_methods == 1:
            axes = [axes]

        for ax, (mkey, mlabel) in zip(axes, methods):
            for split, color in SPLIT_COLORS.items():
                sd = [m for m in all_metrics if m.get("split") == split]
                stds = [m[f"{fs}_{mkey}_std_mean"] for m in sd if f"{fs}_{mkey}_std_mean" in m]
                errs = [m[f"{fs}_error_mean"] for m in sd if f"{fs}_error_mean" in m]
                corrs = [m[f"{fs}_{mkey}_corr"] for m in sd if f"{fs}_{mkey}_corr" in m]
                if not stds:
                    continue
                avg_r = np.mean(corrs) if corrs else 0
                ax.scatter(
                    stds,
                    errs,
                    c=color,
                    s=40,
                    edgecolors="k",
                    linewidths=0.5,
                    alpha=0.7,
                    label=f"{split} (n={len(stds)}, r={avg_r:.2f})",
                )

            # y=x
            all_v = [m.get(f"{fs}_{mkey}_std_mean", 0) for m in all_metrics] + [
                m.get(f"{fs}_error_mean", 0) for m in all_metrics
            ]
            if all_v and max(all_v) > 0:
                lim = max(all_v) * 1.1
                ax.plot([0, lim], [0, lim], "r--", lw=1.5, alpha=0.5, label="y = x")

            ax.set_xlabel(f"{mlabel} σ")
            ax.set_ylabel("Mean |error|")
            ax.set_title(f"{mlabel} — {fs}")
            ax.legend(fontsize=8)

        plt.suptitle(f"Uncertainty vs Error — {fs}", fontsize=13)
        plt.tight_layout()
        fig.savefig(out_dir / f"scatter_{fs}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: scatter_{fs}.png")


def plot_correlation_summary(all_metrics, out_dir, is_uq):
    """Bar chart: correlation per field × method, test vs val."""
    methods = [("shared", "Shared"), ("pf", "Per-field")]
    if is_uq:
        methods.append(("ale", "Aleatoric"))
    fields = ["pressure", "friction"]

    fig, axes = plt.subplots(1, len(fields), figsize=(6 * len(fields), 5))
    if len(fields) == 1:
        axes = [axes]

    for ax, fs in zip(axes, fields):
        x = np.arange(len(methods))
        width = 0.35
        for i, (split, color) in enumerate(SPLIT_COLORS.items()):
            sd = [m for m in all_metrics if m.get("split") == split]
            means, errs = [], []
            for mkey, _ in methods:
                corrs = [m[f"{fs}_{mkey}_corr"] for m in sd if f"{fs}_{mkey}_corr" in m]
                means.append(np.mean(corrs) if corrs else 0)
                errs.append(np.std(corrs) if corrs else 0)
            ax.bar(x + i * width, means, width, yerr=errs, capsize=3, label=split, color=color)

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([ml for _, ml in methods], fontsize=9)
        ax.set_ylabel("Pearson r (σ vs error)")
        ax.set_title(fs)
        ax.legend()
        ax.set_ylim(-0.1, 0.7)
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    plt.suptitle("Correlation Summary: Test vs Val", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / "correlation_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: correlation_summary.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--label", type=str, default=None)
    ap.add_argument("--checkpoint", type=str, default="best")
    ap.add_argument("--num-train-samples", type=int, default=400)
    ap.add_argument("--max-per-split", type=int, default=None, help="Limit per split (default: all)")
    ap.add_argument("--output-dir", type=str, default="outputs/laplace_full")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    label = a.label or run_dir.name
    out = Path(a.output_dir) / label
    out.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"Full Laplace + Aleatoric Evaluation: {label}")
    print(f"{'=' * 60}")

    model, datasets, pipelines, config, is_uq, norms, pos_norm = load_all(run_dir, a.checkpoint, a.device)
    bb = get_backbone(model)
    print(f"  Model: {'UQ' if is_uq else 'Baseline'}, D={bb.surface_decoder.project.in_features}")

    print(f"\n  Fisher ({a.num_train_samples} training samples)...")
    H_inv = build_fisher(model, bb, datasets["train"], pipelines["train"], a.device, a.num_train_samples)
    torch.save({"H_inv": H_inv}, out / "fisher.pt")

    run_ids = get_run_ids(datasets.get("test", datasets.get("val")))
    all_metrics = []

    for split in ["test", "val"]:
        if split not in datasets:
            continue
        ids = run_ids.get(split, [])
        n = min(a.max_per_split, len(ids)) if a.max_per_split else len(ids)
        print(f"\n  --- {split.upper()}: {n} samples ---")

        for i in range(n):
            print(f"\n  [{split}] {i + 1}/{n} (run_{ids[i]})")
            m = process_sample(
                model, bb, datasets[split], pipelines[split], ids[i], i, H_inv, a.device, norms, pos_norm, is_uq, split
            )
            if m:
                all_metrics.append(m)
                parts = []
                for mkey in ["shared", "pf", "ale"]:
                    pc = m.get(f"pressure_{mkey}_corr")
                    fc = m.get(f"friction_{mkey}_corr")
                    if pc is not None or fc is not None:
                        parts.append(f"{mkey}: p={pc:.3f}" + (f" f={fc:.3f}" if fc else ""))
                print(f"    {', '.join(parts)}")

    with open(out / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n  Plots...")
    plot_scatter_all(all_metrics, out, is_uq)
    plot_correlation_summary(all_metrics, out, is_uq)

    # Print summary
    print(f"\n  {'=' * 50}")
    print("  SUMMARY")
    print(f"  {'=' * 50}")
    for split in ["test", "val"]:
        sd = [m for m in all_metrics if m["split"] == split]
        if not sd:
            continue
        print(f"  {split} ({len(sd)} samples):")
        for fs in ["pressure", "friction"]:
            for mkey in ["shared", "pf", "ale"]:
                corrs = [m[f"{fs}_{mkey}_corr"] for m in sd if f"{fs}_{mkey}_corr" in m]
                if corrs:
                    print(f"    {fs:10s} {mkey:8s}: r = {np.mean(corrs):.3f} ± {np.std(corrs):.3f}")

    print(f"\n  Results: {out}")


if __name__ == "__main__":
    main()
