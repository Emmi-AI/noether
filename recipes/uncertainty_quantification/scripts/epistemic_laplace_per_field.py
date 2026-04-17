#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Epistemic uncertainty via last-layer Laplace — per-field output variance.

Instead of a shared feature-space proxy (v1: sigma = ||W|| * sqrt(h^T H^-1 h)),
this computes the PROPER per-field output variance:

    For output dimension j with decoder weight w_j:
        Var[y_j](x) = (w_j^T @ H^{-1} @ h(x))^2

    For a multi-dim field (e.g. friction with 3 components):
        Var_field(x) = sum_j (w_j^T @ H^{-1} @ h(x))^2

This gives DIFFERENT spatial uncertainty patterns for pressure vs friction
because different decoder weights project h into different directions.

Usage:
    uv run python recipes/uncertainty_quantification/scripts/epistemic_laplace_per_field.py \
        --run-dir outputs/<run_id> \
        --label baseline_laplace_pf --num-train-samples 400 --num-test-samples 1
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


def load_model_and_datasets(run_dir, checkpoint, device):
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

    def make_ds_and_pipeline(ds_config):
        ds = DatasetFactory().instantiate(ds_config)
        pl = (
            Factory().create(ds_config.pipeline)
            if ds_config.pipeline
            else __import__("noether.data.pipeline", fromlist=["MultiStagePipeline"]).MultiStagePipeline()
        )
        ds.pipeline = pl
        return ds, pl

    train_ds, train_pl = make_ds_and_pipeline(vc.datasets["train"])
    test_ds, test_pl = make_ds_and_pipeline(vc.datasets["test"])

    inner = test_ds
    while hasattr(inner, "_dataset"):
        inner = inner._dataset
    normalizers = inner.normalizers if hasattr(inner, "normalizers") else {}

    is_uq = hasattr(model, "forward_with_epistemic")
    return model, train_ds, train_pl, test_ds, test_pl, config, is_uq, normalizers, normalizers.get("surface_position")


def get_backbone(model):
    if hasattr(model, "ab_upt"):
        return model.ab_upt
    b = getattr(model, "backbone", None)
    if b and hasattr(b, "backbone"):
        return b.backbone
    if b:
        return b
    raise ValueError(f"No backbone in {type(model).__name__}")


def get_test_run_ids(ds):
    inner = ds
    while hasattr(inner, "_dataset"):
        inner = inner._dataset
    return sorted(inner.get_dataset_splits.test)


def load_mesh(rid):
    p = SURFACE_VTP_ROOT / f"run_{rid}" / f"boundary_{rid}.vtp"
    return pv.read(str(p)) if p.exists() else None


def denorm(t, field, norms):
    return norms[field].inverse(t.cpu()) if field in norms else t


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

    for i in range(min(n_samples, len(ds))):
        if (i + 1) % 10 == 0 or i == min(n_samples, len(ds)) - 1:
            print(f"    Fisher: {i + 1}/{min(n_samples, len(ds))}", end="\r")
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
    return H, torch.linalg.inv(H)


# ---------------------------------------------------------------------------
# Per-field Laplace query inference
# ---------------------------------------------------------------------------


def laplace_per_field(model, backbone, fwd_batch, qpos, H_inv, device, norms, is_uq, chunk_size=CHUNK_SIZE):
    hook = FeatureHook(backbone)
    W = backbone.surface_decoder.project.weight.detach().cpu().double()
    slices = backbone.data_specs.surface_output_dims.field_slices
    H_inv_d = H_inv.double()

    # Precompute W_field @ H_inv per field
    WH = {}
    for field in SURFACE_GT_MAP:
        fn = field.replace("surface_", "")
        s = slices.get(fn)
        WH[field] = (W[s] if s else W) @ H_inv_d  # (D_field, D_hidden)

    n = qpos.shape[0]
    n_ch = max(1, (n + chunk_size - 1) // chunk_size)
    preds: dict[str, list] = defaultdict(list)
    epi: dict[str, list] = defaultdict(list)

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
            for field, wh in WH.items():
                proj = h @ wh.T  # (nq, D_field)
                epi[field].append(torch.sqrt((proj**2).sum(dim=-1).float().clamp(min=0)))

        if (i + 1) % 100 == 0 or i == n_ch - 1:
            print(f"      Chunk {i + 1}/{n_ch}", end="\r")

    print()
    hook.remove()

    preds = {k: torch.cat(v, dim=1) for k, v in preds.items()}
    epi = {k: torch.cat(v, dim=0) for k, v in epi.items()}

    result = {}
    for field in SURFACE_GT_MAP:
        pk = f"{field}_mean" if is_uq else field
        if pk in preds:
            result[f"pred_{field}"] = denorm(preds[pk], field, norms)[0].numpy().squeeze()
        if field in epi:
            one, zero = torch.ones(1, 1, 1), torch.zeros(1, 1, 1)
            sc = (norms[field].inverse(one) - norms[field].inverse(zero)).abs().mean().item() if field in norms else 1.0
            result[f"epistemic_std_{field}"] = (epi[field] * sc).numpy()
        if is_uq and f"{field}_log_var" in preds:
            sn = torch.exp(0.5 * preds[f"{field}_log_var"])
            if field in norms:
                o, z = torch.ones_like(sn), torch.zeros_like(sn)
                ale = (sn.cpu() * (norms[field].inverse(o.cpu()) - norms[field].inverse(z.cpu())).abs())[0].numpy()
            else:
                ale = sn[0].cpu().numpy()
            if ale.ndim > 1 and ale.shape[-1] > 1:
                ale = np.linalg.norm(ale, axis=-1)
            result[f"aleatoric_std_{field}"] = ale.squeeze()

    return result


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def render(mesh, out_dir, idx, is_uq):
    pv.OFF_SCREEN = True
    for field, gt_key in SURFACE_GT_MAP.items():
        fs = field.replace("surface_", "")
        gtk = f"gt_{fs}_mag" if f"gt_{fs}_mag" in mesh.cell_data else gt_key
        pk = f"pred_{fs}_mag" if f"pred_{fs}_mag" in mesh.cell_data else f"pred_{fs}"
        ek, epk, ak = f"error_{fs}", f"laplace_std_{fs}", f"aleatoric_std_{fs}"

        clim = (
            [float(np.percentile(mesh.cell_data[gtk], 1)), float(np.percentile(mesh.cell_data[gtk], 99))]
            if gtk in mesh.cell_data
            else None
        )

        panels = []
        if gtk in mesh.cell_data:
            panels.append(("Ground Truth", gtk, "coolwarm", clim))
        if pk in mesh.cell_data:
            panels.append(("Prediction", pk, "coolwarm", clim))
        if ek in mesh.cell_data:
            d = mesh.cell_data[ek]
            panels.append(("|Error|", ek, "Reds", [0, float(np.percentile(d, 95))]))
        if epk in mesh.cell_data:
            d = mesh.cell_data[epk]
            panels.append(("Laplace σ", epk, "Reds", [float(np.percentile(d, 5)), float(np.percentile(d, 95))]))
        if is_uq and ak in mesh.cell_data:
            d = mesh.cell_data[ak]
            panels.append(("Aleatoric σ", ak, "Reds", [float(np.percentile(d, 5)), float(np.percentile(d, 95))]))
        if not panels:
            continue

        imgs = []
        for title, arr, cmap, cl in panels:
            p = pv.Plotter(off_screen=True, window_size=[600, 500])
            p.add_mesh(
                mesh.copy(),
                scalars=arr,
                cmap=cmap,
                clim=cl,
                show_scalar_bar=True,
                scalar_bar_args={"title": title, "n_labels": 5},
            )
            p.add_text(title, font_size=12, position="upper_left")
            p.camera_position = "xy"
            p.camera.zoom(1.5)
            imgs.append(p.screenshot(return_img=True))
            p.close()

        fig, axes = plt.subplots(1, len(imgs), figsize=(6 * len(imgs), 5))
        if len(imgs) == 1:
            axes = [axes]
        for ax, im in zip(axes, imgs):
            ax.imshow(im)
            ax.axis("off")
        plt.tight_layout(pad=0.5)
        fig.savefig(out_dir / f"laplace_pf_{fs}_sample{idx}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Rendered: laplace_pf_{fs}_sample{idx}.png")


def scatter(mesh, out_dir, idx, is_uq):
    for field, _ in SURFACE_GT_MAP.items():
        fs = field.replace("surface_", "")
        ek, epk, ak = f"error_{fs}", f"laplace_std_{fs}", f"aleatoric_std_{fs}"
        if ek not in mesh.cell_data or epk not in mesh.cell_data:
            continue
        err, ep = mesh.cell_data[ek], mesh.cell_data[epk]

        nc = 2 if (is_uq and ak in mesh.cell_data) else 1
        fig, axes = plt.subplots(1, nc, figsize=(6 * nc, 5))
        if nc == 1:
            axes = [axes]

        for ax_i, (data, lbl, col) in enumerate(
            [(ep, "Laplace σ", "C0")] + ([(mesh.cell_data[ak], "Aleatoric σ", "orange")] if nc == 2 else [])
        ):
            ax = axes[ax_i]
            bins = np.percentile(data, np.linspace(0, 100, 31))
            bc, be = [], []
            for j in range(30):
                m = (data >= bins[j]) & (data < bins[j + 1])
                if m.sum() > 100:
                    bc.append(data[m].mean())
                    be.append(err[m].mean())
            ax.scatter(bc, be, s=30, edgecolors="k", linewidths=0.5, color=col)
            if bc:
                lim = max(max(bc), max(be)) * 1.1
                ax.plot([0, lim], [0, lim], "r--", lw=2, label="y = x")
            ax.set_xlabel(f"{lbl} (binned)")
            ax.set_ylabel("Mean |error|")
            ax.set_title(f"{lbl} vs Error — {fs}")
            ax.legend()

        plt.tight_layout()
        fig.savefig(out_dir / f"laplace_pf_scatter_{fs}_sample{idx}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: laplace_pf_scatter_{fs}_sample{idx}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--label", type=str, default=None)
    ap.add_argument("--checkpoint", type=str, default="best")
    ap.add_argument("--num-train-samples", type=int, default=400)
    ap.add_argument("--num-test-samples", type=int, default=1)
    ap.add_argument("--output-dir", type=str, default="outputs/epistemic_laplace_per_field")
    ap.add_argument("--reg", type=float, default=1e-3, help="Regularization lambda for Fisher (default 1e-3)")
    ap.add_argument("--save-vtp", action="store_true")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    label = a.label or run_dir.name
    out = Path(a.output_dir) / label
    out.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}\nPer-Field Laplace: {label}\n{'=' * 60}")
    model, tr_ds, tr_pl, te_ds, te_pl, cfg, is_uq, norms, pos_norm = load_model_and_datasets(
        run_dir, a.checkpoint, a.device
    )
    bb = get_backbone(model)
    print(f"  Model: {'UQ' if is_uq else 'Baseline'}, D={bb.surface_decoder.project.in_features}")

    print(f"\n  Fisher ({a.num_train_samples} samples, reg={a.reg})...")
    H, Hi = build_fisher(model, bb, tr_ds, tr_pl, a.device, a.num_train_samples, reg=a.reg)
    torch.save({"H": H, "H_inv": Hi}, out / "fisher.pt")

    ids = get_test_run_ids(te_ds)
    all_sample_metrics = []
    for i in range(min(a.num_test_samples, len(te_ds))):
        print(f"\n  Test {i + 1} (run_{ids[i]})")
        mesh = load_mesh(ids[i])
        if not mesh:
            continue

        cc = torch.tensor(mesh.cell_centers().points, dtype=torch.float32)
        qp = pos_norm(cc) if pos_norm else cc

        batch = te_pl([te_ds[i]])
        batch = {k: v.to(a.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        fwd = {k: v for k, v in batch.items() if k in FORWARD_PROPERTIES}

        print(f"    Laplace per-field at {mesh.n_cells} pts...")
        res = laplace_per_field(model, bb, fwd, qp, Hi, a.device, norms, is_uq)

        metrics = {}
        for field, gtk in SURFACE_GT_MAP.items():
            if gtk not in mesh.cell_data:
                continue
            gt = mesh.cell_data[gtk]
            fs = field.replace("surface_", "")
            pred, ep = res.get(f"pred_{field}"), res.get(f"epistemic_std_{field}")
            if pred is None:
                continue

            if pred.ndim > 1 and pred.shape[-1] > 1:
                mesh.cell_data[f"pred_{fs}_mag"] = np.linalg.norm(pred, axis=-1)
                mesh.cell_data[f"gt_{fs}_mag"] = np.linalg.norm(gt, axis=-1)
                mesh.cell_data[f"error_{fs}"] = np.linalg.norm(pred - gt, axis=-1)
            else:
                mesh.cell_data[f"pred_{fs}"] = pred.squeeze()
                mesh.cell_data[f"error_{fs}"] = np.abs(pred.squeeze() - gt.squeeze())
            if ep is not None:
                mesh.cell_data[f"laplace_std_{fs}"] = ep.squeeze()
            ale = res.get(f"aleatoric_std_{field}")
            if ale is not None:
                mesh.cell_data[f"aleatoric_std_{fs}"] = ale.squeeze()

            err = mesh.cell_data[f"error_{fs}"]
            if ep is not None:
                ep_arr = mesh.cell_data[f"laplace_std_{fs}"]
                metrics[f"{fs}_error_mean"] = float(err.mean())
                metrics[f"{fs}_laplace_mean"] = float(ep_arr.mean())
                metrics[f"{fs}_laplace_median"] = float(np.median(ep_arr))
                metrics[f"{fs}_corr"] = float(np.corrcoef(err.ravel(), ep_arr.ravel())[0, 1])

        print("    --- Metrics ---")
        for k, v in sorted(metrics.items()):
            print(f"      {k}: {v:.6f}")

        all_sample_metrics.append(metrics)

        render(mesh, out, i, is_uq)
        scatter(mesh, out, i, is_uq)
        if a.save_vtp:
            mesh.save(str(out / f"sample_{i}.vtp"))
        json.dump(all_sample_metrics, open(out / "metrics.json", "w"), indent=2)

    print(f"\nDone: {out}")


if __name__ == "__main__":
    main()
