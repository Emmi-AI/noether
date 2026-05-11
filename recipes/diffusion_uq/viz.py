#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""visualization utilities for steady diffusion notebooks."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

logging.getLogger("matplotlib").setLevel(logging.WARNING)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_latent_tsne(
    latent_dir: str | Path,
    dataset_root: str | Path,
    n_show: int | None = None,
):
    """t-sne of per-sample latents colored by mean Cp and |WSS|."""
    latent_dir = Path(latent_dir)
    latent_files = sorted(latent_dir.glob("*.pt"))
    latents = torch.stack([torch.load(f, weights_only=True)["latents"] for f in latent_files])
    if n_show:
        latents = latents[:n_show]
        latent_files = latent_files[:n_show]

    data_root = Path(dataset_root)
    run_dirs = sorted([d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("run_")])
    mean_cp, mean_wss = [], []
    for run_dir in run_dirs[: len(latent_files)]:
        cp = torch.load(run_dir / "surface_pressure.pt", weights_only=True)
        wss = torch.load(run_dir / "surface_wallshearstress.pt", weights_only=True)
        mean_cp.append(cp.mean().item())
        mean_wss.append(wss.norm(dim=-1).mean().item())
    mean_cp = np.array(mean_cp)
    mean_wss = np.array(mean_wss)

    flat = latents.reshape(latents.shape[0], -1).numpy()
    tsne = TSNE(n_components=2, perplexity=min(30, len(flat) - 1), random_state=42)
    emb = tsne.fit_transform(flat)

    n_sn = min(50, latents.shape[0])
    sn_flat = latents[:n_sn].reshape(-1, latents.shape[-1]).numpy()
    sn_labels = np.repeat(np.arange(n_sn), latents.shape[1])
    tsne_sn = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_sn = tsne_sn.fit_transform(sn_flat)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    sc1 = axes[0].scatter(emb[:, 0], emb[:, 1], c=mean_cp, cmap="RdBu_r", s=15, alpha=0.8)
    axes[0].set_title("t-sne (color = mean Cp)")
    plt.colorbar(sc1, ax=axes[0], label="mean Cp")

    sc2 = axes[1].scatter(emb[:, 0], emb[:, 1], c=mean_wss, cmap="magma", s=15, alpha=0.8)
    axes[1].set_title("t-sne (color = mean |WSS|)")
    plt.colorbar(sc2, ax=axes[1], label="mean |WSS|")

    axes[2].scatter(emb_sn[:, 0], emb_sn[:, 1], c=sn_labels, cmap="tab20", s=3, alpha=0.5)
    axes[2].set_title(f"t-sne supernode tokens (first {n_sn} samples)")

    plt.tight_layout()
    plt.show()
    return latents, latent_files


def plot_field_scatter(
    positions: np.ndarray,
    original: np.ndarray,
    predicted: np.ndarray,
    field_name: str = "Cp",
    point_size: float = 0.3,
    elev: float = 20,
    azim: float = -60,
):
    """3d scatter: original vs predicted vs error on a car surface."""
    error = original - predicted
    vmin, vmax = original.min(), original.max()
    emax = np.abs(error).max()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw={"projection": "3d"})
    for ax, vals, title, vm in [
        (axes[0], original, f"original {field_name}", (vmin, vmax)),
        (axes[1], predicted, f"predicted {field_name}", (vmin, vmax)),
        (axes[2], error, "error", (-emax, emax)),
    ]:
        sc = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=vals,
            cmap="RdBu_r",
            s=point_size,
            alpha=0.8,
            vmin=vm[0],
            vmax=vm[1],
        )
        ax.set_title(title)
        fig.colorbar(sc, ax=ax, shrink=0.6)
        ax.view_init(elev=elev, azim=azim)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()

    mse = (error**2).mean()
    print(f"{field_name} mse: {mse:.6f} (n={len(original)} points)")
    return mse


def plot_field_uq(
    positions: np.ndarray,
    target: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    field_name: str = "Cp",
    point_size: float = 0.3,
    elev: float = 20,
    azim: float = -60,
    index: int | None = None,
):
    """4-panel: ground truth, predicted (mean), uncertainty (std), error |pred-gt|.

    correlation between uq (std) and error is reported in the title of panel 3/4.
    """
    error = np.abs(target - pred_mean)
    vmin, vmax = float(target.min()), float(target.max())
    emax = float(max(error.max(), 1e-12))
    umax = float(max(pred_std.max(), 1e-12))

    # pearson correlation between uq and error
    eps = 1e-12
    corr = float(np.corrcoef(pred_std.ravel(), error.ravel())[0, 1]) if pred_std.std() > eps else float("nan")

    fig, axes = plt.subplots(1, 4, figsize=(26, 6), subplot_kw={"projection": "3d"})
    panels = [
        (axes[0], target, f"gt {field_name}", "RdBu_r", (vmin, vmax)),
        (axes[1], pred_mean, f"pred mean {field_name}", "RdBu_r", (vmin, vmax)),
        (axes[2], pred_std, f"uq (std), corr={corr:.3f}", "viridis", (0, umax)),
        (axes[3], error, "|pred - gt|", "magma", (0, emax)),
    ]
    for ax, vals, title, cmap, vm in panels:
        sc = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=vals,
            cmap=cmap,
            s=point_size,
            alpha=0.85,
            vmin=vm[0],
            vmax=vm[1],
        )
        ax.set_title(title)
        fig.colorbar(sc, ax=ax, shrink=0.6)
        ax.view_init(elev=elev, azim=azim)
        ax.set_aspect("equal")

    suptitle = f"{field_name} — sample {index}" if index is not None else field_name
    fig.suptitle(suptitle, fontsize=14, x=0.5, ha="center")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
    return corr


def plot_field_uq_front(
    positions: np.ndarray,
    target: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    field_name: str = "Cp",
    index: int | None = None,
    x_window: float | None = None,
):
    """Continuous front-of-car view via Delaunay triangulation on the (Y,Z) plane.

    The car's longitudinal axis is X, so looking from the front = projecting to
    (Y, Z). Triangulating that projection and coloring with ``tripcolor`` gives
    a continuous interpolated surface, much more readable than a 3D scatter.

    ``x_window``: optional — keep only points with X within the frontmost
    ``x_window`` fraction of the range (e.g., 0.25 = front quarter). ``None``
    uses all points.
    """
    import matplotlib.tri as mtri

    pts = positions
    values = dict(target=target, pred_mean=pred_mean, pred_std=pred_std)
    if x_window is not None:
        x = pts[:, 0]
        x_min, x_max = float(x.min()), float(x.max())
        cutoff = x_min + x_window * (x_max - x_min)
        mask = x <= cutoff
        pts = pts[mask]
        values = {k: v[mask] for k, v in values.items()}

    y, z = pts[:, 1], pts[:, 2]
    tri = mtri.Triangulation(y, z)

    error = np.abs(values["target"] - values["pred_mean"])
    vmin, vmax = float(values["target"].min()), float(values["target"].max())
    umax = float(max(values["pred_std"].max(), 1e-12))
    emax = float(max(error.max(), 1e-12))

    eps = 1e-12
    corr = (
        float(np.corrcoef(values["pred_std"].ravel(), error.ravel())[0, 1])
        if values["pred_std"].std() > eps
        else float("nan")
    )

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    panels = [
        (axes[0], values["target"], f"gt {field_name}", "RdBu_r", vmin, vmax),
        (axes[1], values["pred_mean"], f"pred mean {field_name}", "RdBu_r", vmin, vmax),
        (axes[2], values["pred_std"], f"uq (std), corr={corr:.3f}", "viridis", 0, umax),
        (axes[3], error, "|pred - gt|", "magma", 0, emax),
    ]
    for ax, vals, title, cmap, vmn, vmx in panels:
        tpc = ax.tripcolor(tri, vals, cmap=cmap, vmin=vmn, vmax=vmx, shading="gouraud")
        ax.set_title(title)
        ax.set_xlabel("Y")
        ax.set_ylabel("Z")
        ax.set_aspect("equal")
        fig.colorbar(tpc, ax=ax, shrink=0.8)

    suptitle = f"{field_name} (front view) — sample {index}" if index is not None else f"{field_name} (front view)"
    fig.suptitle(suptitle, fontsize=14, x=0.5, ha="center")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
    return corr


def plot_field_uq_mesh(
    positions: np.ndarray,
    target: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    field_name: str = "Cp",
    index: int | None = None,
    elev: float = 5,
    azim: float = 180,
    edge_quantile: float = 0.98,
):
    """4-panel 3D **triangulated surface** view of the car from the front.

    Builds a 2-D Delaunay triangulation on the (Y, Z) plane (front view), then
    renders the full 3-D surface with ``plot_trisurf`` and colors faces by the
    scalar field. Much more readable than a 3D scatter because the car's
    silhouette and surface gradients are preserved.

    ``edge_quantile``: drop triangles whose longest (Y,Z) edge is above this
    quantile — removes the spurious "skin" that Delaunay draws across sparse
    regions / concave boundaries. Default 0.98 keeps 98% of triangles.
    ``elev``, ``azim``: camera. Default = front-facing looking down +X axis.
    """
    import matplotlib.tri as mtri

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    triang = mtri.Triangulation(y, z)

    # filter long triangles (hull artifacts across sparse regions)
    tris = triang.triangles
    edges = np.stack(
        [
            np.linalg.norm(positions[tris[:, 0], 1:] - positions[tris[:, 1], 1:], axis=1),
            np.linalg.norm(positions[tris[:, 1], 1:] - positions[tris[:, 2], 1:], axis=1),
            np.linalg.norm(positions[tris[:, 2], 1:] - positions[tris[:, 0], 1:], axis=1),
        ],
        axis=1,
    ).max(axis=1)
    mask = edges > np.quantile(edges, edge_quantile)
    triang.set_mask(mask)

    error = np.abs(target - pred_mean)
    vmin, vmax = float(target.min()), float(target.max())
    umax = float(max(pred_std.max(), 1e-12))
    emax = float(max(error.max(), 1e-12))

    eps = 1e-12
    corr = float(np.corrcoef(pred_std.ravel(), error.ravel())[0, 1]) if pred_std.std() > eps else float("nan")

    fig, axes = plt.subplots(1, 4, figsize=(26, 7), subplot_kw={"projection": "3d"})
    panels = [
        (axes[0], target, f"gt {field_name}", "RdBu_r", vmin, vmax),
        (axes[1], pred_mean, f"pred mean {field_name}", "RdBu_r", vmin, vmax),
        (axes[2], pred_std, f"uq (std), corr={corr:.3f}", "viridis", 0, umax),
        (axes[3], error, "|pred - gt|", "magma", 0, emax),
    ]

    for ax, vals, title, cmap, vmn, vmx in panels:
        # per-face color = mean over triangle vertices; robust and fast
        face_vals = vals[tris].mean(axis=1)
        collec = ax.plot_trisurf(
            triang,
            x,
            cmap=cmap,
            linewidth=0,
            antialiased=False,
            shade=False,
        )
        collec.set_array(face_vals)
        collec.set_clim(vmn, vmx)
        ax.set_title(title)
        fig.colorbar(collec, ax=ax, shrink=0.55, pad=0.02)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
        ax.set_axis_off()

    suptitle = (
        f"{field_name} (front view, triangulated) — sample {index}"
        if index is not None
        else f"{field_name} (front view, triangulated)"
    )
    fig.suptitle(suptitle, fontsize=14, x=0.5, ha="center")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
    return corr


def plot_field_uq_stl(
    positions: np.ndarray,
    target: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    stl_path: str | Path,
    field_name: str = "Cp",
    index: int | None = None,
    view: str = "front",
    angle: float = 30.0,
    elevation: float = 10.0,
    window_size: tuple[int, int] = (600, 500),
    interp_k: int = 16,
    kernel: str = "gauss",
    align: str = "bbox",
    pos_normalizer=None,
):
    """4-panel car view using the **real STL** colormapped by nearest anchor.

    Loads the raw DrivAerML STL mesh, KD-tree-interpolates anchor scalars onto
    the STL vertices, and renders 4 pyvista off-screen screenshots which are
    composited into a single matplotlib figure. This is the topologically
    correct way to view surface fields — no fake triangles across the car.

    ``stl_path``: e.g.
        /nfs-gpu/research/datasets/drivaerml/raw_surface_data/run_<N>/drivaer_<N>.stl
    ``view``: one of {"front", "rear", "side", "top", "iso"}.
    ``kernel``: "gauss" (per-query bandwidth = mean of k distances) or "idw"
        (inverse distance squared).
    ``align``: "bbox" matches anchor cloud bbox to STL bbox (default; works
        even with no normalizer info but biased when sample cloud underfills
        true bounds). "normalizer" inverts ``pos_normalizer`` on anchor
        positions to get true physical coords — exact, but requires passing
        the dataset's ``PositionNormalizer``.
    """
    import pyvista as pv
    import torch
    from scipy.spatial import cKDTree

    mesh = pv.read(str(stl_path))

    if align == "normalizer":
        assert pos_normalizer is not None, "align='normalizer' requires pos_normalizer"
        pos_t = positions if torch.is_tensor(positions) else torch.as_tensor(positions, dtype=torch.float32)
        anchors_phys = pos_normalizer.inverse(pos_t).cpu().numpy()
        query_pts = mesh.points  # STL already in raw physical (mm)
        kdt_pts = anchors_phys
    else:
        # bbox alignment fallback. Anchors come out of the pipeline
        # position-normalized; STL is raw mm. Match bboxes (biased when the
        # sample cloud underfills the true mesh extent).
        a_min = positions.min(axis=0)
        a_max = positions.max(axis=0)
        s_min = mesh.points.min(axis=0)
        s_max = mesh.points.max(axis=0)
        denom = np.where(s_max - s_min > 1e-9, s_max - s_min, 1.0)
        scale = (a_max - a_min) / denom
        shift = a_min - s_min * scale
        query_pts = mesh.points * scale + shift
        kdt_pts = positions

    kdt = cKDTree(kdt_pts)
    dists, idxs = kdt.query(query_pts, k=max(interp_k, 1))
    if interp_k == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    if kernel == "gauss":
        # per-query bandwidth = mean of k distances → adapts to local density
        sigma = dists.mean(axis=1, keepdims=True) + 1e-12
        weights = np.exp(-0.5 * (dists / sigma) ** 2)
    else:
        weights = 1.0 / (dists**2 + 1e-12)
    weights /= weights.sum(axis=1, keepdims=True)

    def interp(field: np.ndarray) -> np.ndarray:
        return (field[idxs] * weights).sum(axis=1)

    error = np.abs(target - pred_mean)
    vmin, vmax = float(target.min()), float(target.max())
    # robust upper clip: 98th percentile avoids a handful of outlier points
    # crushing the rest of the field into the darkest colour band
    umax = float(max(np.percentile(pred_std, 98), 1e-12))
    emax = float(max(np.percentile(error, 98), 1e-12))

    eps = 1e-12
    corr = float(np.corrcoef(pred_std.ravel(), error.ravel())[0, 1]) if pred_std.std() > eps else float("nan")

    mesh["gt"] = interp(target)
    mesh["pred"] = interp(pred_mean)
    mesh["std"] = interp(pred_std)
    mesh["err"] = interp(error)

    # bright sequential cmaps for std / error so hotspots pop against a light
    # background instead of being buried in dark blue/black
    panels = [
        ("gt", f"gt {field_name}", "RdBu_r", (vmin, vmax)),
        ("pred", f"pred mean {field_name}", "RdBu_r", (vmin, vmax)),
        ("std", f"uq (std), corr={corr:.3f}", "YlOrRd", (0, umax)),
        ("err", "|pred - gt|", "Reds", (0, emax)),
    ]

    # camera placement around mesh bounds. DrivAerML convention: car faces -X
    # (nose at xmin), so "front" means camera on the -X side looking toward +X.
    # ``angle`` rotates camera around Z by that many degrees off head-on
    # (positive = toward +Y). ``elevation`` tilts the camera up by that many
    # degrees above the XY plane.
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    span = max(xmax - xmin, ymax - ymin, zmax - zmin) * 1.3
    import math

    th = math.radians(angle)
    el = math.radians(elevation)
    if view == "front":
        base_dir = (-math.cos(th) * math.cos(el), math.sin(th) * math.cos(el), math.sin(el))
    elif view == "rear":
        base_dir = (math.cos(th) * math.cos(el), math.sin(th) * math.cos(el), math.sin(el))
    elif view == "side":
        base_dir = (math.sin(th) * math.cos(el), math.cos(th) * math.cos(el), math.sin(el))
    elif view == "top":
        base_dir = (0.0, 0.0, 1.0)
    else:  # iso
        base_dir = (-0.7, 0.7, 0.3)
    cam_pos = (cx + base_dir[0] * span, cy + base_dir[1] * span, cz + base_dir[2] * span)
    up = (0, 0, 1) if view != "top" else (1, 0, 0)
    camera = [cam_pos, (cx, cy, cz), up]

    images = []
    for scalar, _title, cmap, clim in panels:
        p = pv.Plotter(off_screen=True, window_size=window_size)
        p.add_mesh(mesh, scalars=scalar, cmap=cmap, clim=clim, show_scalar_bar=False)
        p.set_background("white")
        p.camera_position = camera
        images.append(p.screenshot(return_img=True))
        p.close()

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    for ax, img, (_, title, cmap, clim) in zip(axes, images, panels, strict=True):
        ax.imshow(img)
        ax.set_title(title)
        ax.set_axis_off()
        norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6)

    suptitle = f"{field_name} ({view} view) — sample {index}" if index is not None else f"{field_name} ({view} view)"
    fig.suptitle(suptitle, fontsize=14, x=0.5, ha="center")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
    return corr


@torch.no_grad()
def decode_on_stl_vertices(
    ae_model,
    latents: Tensor,
    stl_path: str | Path,
    anchor_positions: np.ndarray,
    volume_anchor_position: Tensor,
    geometry_position: Tensor | None = None,
    geometry_supernode_idx: Tensor | None = None,
    chunk_size: int = 8192,
    surface_fields: tuple[str, ...] = (
        "surface_pressure",
        "surface_friction",
    ),
):
    """Query the AE decoder at every STL vertex (chunked) → dense per-vertex fields.

    Uses the anchor-vs-STL bbox-matching trick to map STL raw mm into the
    anchor's normalized coordinate frame before passing as query positions.

    Args:
        ae_model: an AE with ``decode(latents, surface_anchor_position,
            volume_anchor_position, geometry_position, geometry_supernode_idx)``
            (ABUPTAutoencoder).
        latents: ``(1, n_tokens, latent_dim)`` — AE's latent code for the sample.
        stl_path: path to the raw STL.
        anchor_positions: the (N, 3) anchor positions from ``batch`` in
            normalized space, used only for bbox alignment.
        volume_anchor_position / geometry_*: forwarded unchanged to ``decode``.
        chunk_size: decoder call size along the vertex dimension.
        surface_fields: which surface fields to collect; volume fields are
            discarded since they're not defined on the STL surface.

    Returns ``(pyvista_mesh, {field_name: (n_verts,) or (n_verts, c) np.ndarray})``.
    """
    import pyvista as pv

    mesh = pv.read(str(stl_path))

    a_min = anchor_positions.min(axis=0)
    a_max = anchor_positions.max(axis=0)
    s_min = mesh.points.min(axis=0)
    s_max = mesh.points.max(axis=0)
    denom = np.where(s_max - s_min > 1e-9, s_max - s_min, 1.0)
    scale = (a_max - a_min) / denom
    shift = a_min - s_min * scale
    device = latents.device
    verts_aligned = torch.from_numpy((mesh.points * scale + shift).astype(np.float32)).to(device)

    n_verts = verts_aligned.shape[0]
    per_field_chunks: dict[str, list[np.ndarray]] = {f: [] for f in surface_fields}
    for i in range(0, n_verts, chunk_size):
        chunk = verts_aligned[i : i + chunk_size].unsqueeze(0)  # (1, k, 3)
        out = ae_model.decode(
            latents,
            chunk,
            volume_anchor_position,
            geometry_position,
            geometry_supernode_idx,
        )
        for f in surface_fields:
            if f in out:
                per_field_chunks[f].append(out[f][0].cpu().numpy())

    dense = {f: np.concatenate(chunks, axis=0) for f, chunks in per_field_chunks.items() if chunks}
    return mesh, dense


def plot_field_dense_stl(
    mesh,
    target_dense: np.ndarray,
    pred_mean_dense: np.ndarray,
    pred_std_dense: np.ndarray | None = None,
    field_name: str = "Cp",
    index: int | None = None,
    view: str = "front",
    angle: float = 30.0,
    elevation: float = 10.0,
    window_size: tuple[int, int] = (600, 500),
):
    """Render pre-decoded per-vertex fields on an STL mesh (no interpolation).

    If ``pred_std_dense`` is provided, produces a 4-panel UQ view (gt / pred /
    std / |err|); otherwise a 3-panel reconstruction view (gt / pred / |err|).

    ``mesh``: ``pv.PolyData`` returned by ``decode_on_stl_vertices`` (or any
    pyvista mesh of matching vertex count).
    """
    import math

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import pyvista as pv

    error = np.abs(target_dense - pred_mean_dense)
    vmin, vmax = float(target_dense.min()), float(target_dense.max())
    emax = float(max(error.max(), 1e-12))

    mesh = mesh.copy()  # don't mutate caller's mesh
    mesh["gt"] = target_dense
    mesh["pred"] = pred_mean_dense
    mesh["err"] = error

    panels = [
        ("gt", f"gt {field_name}", "RdBu_r", (vmin, vmax)),
        ("pred", f"pred {field_name}", "RdBu_r", (vmin, vmax)),
    ]
    if pred_std_dense is not None:
        umax = float(max(pred_std_dense.max(), 1e-12))
        eps = 1e-12
        corr = (
            float(np.corrcoef(pred_std_dense.ravel(), error.ravel())[0, 1])
            if pred_std_dense.std() > eps
            else float("nan")
        )
        mesh["std"] = pred_std_dense
        panels.append(("std", f"uq (std), corr={corr:.3f}", "viridis", (0, umax)))
    panels.append(("err", "|pred - gt|", "magma", (0, emax)))

    # camera (same conventions as plot_field_uq_stl)
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    span = max(xmax - xmin, ymax - ymin, zmax - zmin) * 1.3
    th = math.radians(angle)
    el = math.radians(elevation)
    if view == "front":
        base_dir = (-math.cos(th) * math.cos(el), math.sin(th) * math.cos(el), math.sin(el))
    elif view == "rear":
        base_dir = (math.cos(th) * math.cos(el), math.sin(th) * math.cos(el), math.sin(el))
    elif view == "side":
        base_dir = (math.sin(th) * math.cos(el), math.cos(th) * math.cos(el), math.sin(el))
    elif view == "top":
        base_dir = (0.0, 0.0, 1.0)
    else:
        base_dir = (-0.7, 0.7, 0.3)
    cam_pos = (cx + base_dir[0] * span, cy + base_dir[1] * span, cz + base_dir[2] * span)
    up = (0, 0, 1) if view != "top" else (1, 0, 0)
    camera = [cam_pos, (cx, cy, cz), up]

    images = []
    for scalar, _title, cmap, clim in panels:
        p = pv.Plotter(off_screen=True, window_size=window_size)
        p.add_mesh(mesh, scalars=scalar, cmap=cmap, clim=clim, show_scalar_bar=False)
        p.set_background("white")
        p.camera_position = camera
        images.append(p.screenshot(return_img=True))
        p.close()

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6))
    if len(panels) == 1:
        axes = [axes]
    for ax, img, (_, title, cmap, clim) in zip(axes, images, panels, strict=True):
        ax.imshow(img)
        ax.set_title(title)
        ax.set_axis_off()
        norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6)

    kind = "recon" if pred_std_dense is None else "UQ"
    suptitle = (
        f"{field_name} {kind} ({view} view, dense) — sample {index}"
        if index is not None
        else f"{field_name} {kind} ({view} view, dense)"
    )
    fig.suptitle(suptitle, fontsize=14, x=0.5, ha="center")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


def plot_field_recon_stl(
    positions: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    stl_path: str | Path,
    field_name: str = "Cp",
    index: int | None = None,
    view: str = "front",
    angle: float = 30.0,
    elevation: float = 10.0,
    window_size: tuple[int, int] = (600, 500),
    interp_k: int = 16,
    kernel: str = "gauss",
    align: str = "bbox",
    pos_normalizer=None,
):
    """3-panel reconstruction view (gt / pred / |err|) on the STL.

    Same STL-mapping + k-NN interpolation as ``plot_field_uq_stl``, but
    without the std / UQ panel — intended for AE or regression reconstructions
    where you have a single deterministic prediction. See ``plot_field_uq_stl``
    for ``kernel`` / ``align`` / ``pos_normalizer`` semantics.
    """
    import math

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import pyvista as pv
    import torch
    from scipy.spatial import cKDTree

    mesh = pv.read(str(stl_path))

    if align == "normalizer":
        assert pos_normalizer is not None, "align='normalizer' requires pos_normalizer"
        pos_t = positions if torch.is_tensor(positions) else torch.as_tensor(positions, dtype=torch.float32)
        anchors_phys = pos_normalizer.inverse(pos_t).cpu().numpy()
        query_pts = mesh.points
        kdt_pts = anchors_phys
    else:
        a_min = positions.min(axis=0)
        a_max = positions.max(axis=0)
        s_min = mesh.points.min(axis=0)
        s_max = mesh.points.max(axis=0)
        denom = np.where(s_max - s_min > 1e-9, s_max - s_min, 1.0)
        scale = (a_max - a_min) / denom
        shift = a_min - s_min * scale
        query_pts = mesh.points * scale + shift
        kdt_pts = positions

    kdt = cKDTree(kdt_pts)
    dists, idxs = kdt.query(query_pts, k=max(interp_k, 1))
    if interp_k == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    if kernel == "gauss":
        sigma = dists.mean(axis=1, keepdims=True) + 1e-12
        weights = np.exp(-0.5 * (dists / sigma) ** 2)
    else:
        weights = 1.0 / (dists**2 + 1e-12)
    weights /= weights.sum(axis=1, keepdims=True)

    def interp(field: np.ndarray) -> np.ndarray:
        return (field[idxs] * weights).sum(axis=1)

    error = np.abs(target - pred)
    vmin, vmax = float(target.min()), float(target.max())
    emax = float(max(error.max(), 1e-12))

    mesh["gt"] = interp(target)
    mesh["pred"] = interp(pred)
    mesh["err"] = interp(error)

    panels = [
        ("gt", f"gt {field_name}", "RdBu_r", (vmin, vmax)),
        ("pred", f"pred {field_name}", "RdBu_r", (vmin, vmax)),
        ("err", "|pred - gt|", "magma", (0, emax)),
    ]

    # same camera logic as plot_field_uq_stl
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    span = max(xmax - xmin, ymax - ymin, zmax - zmin) * 1.3
    th = math.radians(angle)
    el = math.radians(elevation)
    if view == "front":
        base_dir = (-math.cos(th) * math.cos(el), math.sin(th) * math.cos(el), math.sin(el))
    elif view == "rear":
        base_dir = (math.cos(th) * math.cos(el), math.sin(th) * math.cos(el), math.sin(el))
    elif view == "side":
        base_dir = (math.sin(th) * math.cos(el), math.cos(th) * math.cos(el), math.sin(el))
    elif view == "top":
        base_dir = (0.0, 0.0, 1.0)
    else:
        base_dir = (-0.7, 0.7, 0.3)
    cam_pos = (cx + base_dir[0] * span, cy + base_dir[1] * span, cz + base_dir[2] * span)
    up = (0, 0, 1) if view != "top" else (1, 0, 0)
    camera = [cam_pos, (cx, cy, cz), up]

    images = []
    for scalar, _title, cmap, clim in panels:
        p = pv.Plotter(off_screen=True, window_size=window_size)
        p.add_mesh(mesh, scalars=scalar, cmap=cmap, clim=clim, show_scalar_bar=False)
        p.set_background("white")
        p.camera_position = camera
        images.append(p.screenshot(return_img=True))
        p.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, img, (_, title, cmap, clim) in zip(axes, images, panels, strict=True):
        ax.imshow(img)
        ax.set_title(title)
        ax.set_axis_off()
        norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6)

    suptitle = (
        f"{field_name} recon ({view} view) — sample {index}"
        if index is not None
        else f"{field_name} recon ({view} view)"
    )
    fig.suptitle(suptitle, fontsize=14, x=0.5, ha="center")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


def plot_generated_vs_real_tsne(
    real_latents: np.ndarray,
    generated_latents: np.ndarray,
):
    """t-sne overlay of real (blue) vs generated (red) latents."""
    n = real_latents.shape[0]
    combined = np.concatenate(
        [
            real_latents.reshape(n, -1),
            generated_latents.reshape(generated_latents.shape[0], -1),
        ]
    )
    tsne = TSNE(n_components=2, perplexity=min(20, n - 1), random_state=42)
    emb = tsne.fit_transform(combined)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(emb[:n, 0], emb[:n, 1], c="tab:blue", s=20, alpha=0.7, label="real")
    ax.scatter(emb[n:, 0], emb[n:, 1], c="tab:red", s=20, alpha=0.7, label="generated")
    ax.legend()
    ax.set_title("t-sne: real vs generated latents")
    plt.show()


def compute_field_metrics(
    pred: dict[str, np.ndarray | Tensor],
    batch: dict[str, np.ndarray | Tensor],
    field_map: dict[str, str] | None = None,
) -> dict[str, float]:
    """compute per-field relative L2 error.

    relL2 = sqrt(sum((p - t)^2) / sum(t^2)) = sqrt(NMSE).

    args:
        pred: model output dict with keys like 'surface_pressure', 'volume_velocity', etc.
        batch: batch dict with keys like 'surface_pressure_target', etc.
        field_map: optional mapping from field_name to target_key. defaults to
            standard DrivAerML fields.

    returns:
        dict of {field_name + "_relL2": float}
    """
    if field_map is None:
        field_map = {
            "surface_pressure": "surface_pressure_target",
            "surface_friction": "surface_friction_target",
            "volume_pressure": "volume_pressure_target",
            "volume_velocity": "volume_velocity_target",
            "volume_vorticity": "volume_vorticity_target",
        }

    metrics: dict[str, float] = {}
    for field_name, target_key in field_map.items():
        if field_name not in pred or target_key not in batch:
            continue
        p = pred[field_name]
        t = batch[target_key]
        if isinstance(p, Tensor):
            p = p.detach().cpu().numpy()
        if isinstance(t, Tensor):
            t = t.detach().cpu().numpy()
        num = float(((p - t) ** 2).sum())
        den = float((t**2).sum())
        metrics[f"{field_name}_relL2"] = float(np.sqrt(num / max(den, 1e-20)))
    return metrics


def evaluate_on_dataset(
    model,
    dataset,
    device: str = "cuda",
    max_samples: int | None = None,
    desc: str = "abupt eval",
) -> dict[str, float]:
    """run autoencoder on dataset samples and compute per-field MSE."""
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    all_metrics: dict[str, list[float]] = {}

    model.eval()
    with torch.no_grad():
        for i in range(n):
            batch = dataset.pipeline([dataset[i]])
            batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}

            pred = model(
                geometry_position=batch["geometry_position"],
                geometry_supernode_idx=batch["geometry_supernode_idx"],
                geometry_batch_idx=batch["geometry_batch_idx"],
                surface_anchor_position=batch["surface_anchor_position"],
                volume_anchor_position=batch["volume_anchor_position"],
                surface_pressure_target=batch.get("surface_pressure_target"),
                surface_friction_target=batch.get("surface_friction_target"),
                volume_pressure_target=batch.get("volume_pressure_target"),
                volume_velocity_target=batch.get("volume_velocity_target"),
                volume_vorticity_target=batch.get("volume_vorticity_target"),
            )

            sample_metrics = compute_field_metrics(pred, batch)
            for k, v in sample_metrics.items():
                all_metrics.setdefault(k, []).append(v)

            if (i + 1) % 10 == 0 or i == n - 1:
                print(f"  [{desc}] {i + 1}/{n}")

    avg = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    avg["total_mse"] = float(np.mean([np.mean(v) for v in all_metrics.values()]))
    return avg


def evaluate_diffusion_on_dataset(
    diff_model,
    ae_model,
    latent_files: list,
    dataset,
    schedule,
    latent_scale: float,
    device: str = "cuda",
    num_samples: int = 1,
    sampling_steps: int = 10,
    max_geoms: int | None = None,
    desc: str = "abupt diffusion eval",
) -> dict[str, float]:
    """sample from diffusion, decode through AE, compute per-field MSE."""
    n = len(latent_files) if max_geoms is None else min(max_geoms, len(latent_files))
    n = min(n, len(dataset))
    all_metrics: dict[str, list[float]] = {}

    diff_model.eval()
    ae_model.eval()

    with torch.no_grad():
        for i in range(n):
            ref = torch.load(latent_files[i], weights_only=True)
            anchor_pos = ref["supernode_positions"].unsqueeze(0).to(device)
            latent_shape = ref["latents"].shape

            sample_latents = []
            for _ in range(num_samples):
                model_fn = lambda x, t, c, sp=anchor_pos: diff_model(x, timestep=t, supernode_positions=sp)
                z = schedule.sample((1, *latent_shape), model_fn, steps=sampling_steps)
                sample_latents.append(z.squeeze(0))

            mean_latent = torch.stack(sample_latents).mean(dim=0).unsqueeze(0)

            # decode through AB-UPT AE
            batch = dataset.pipeline([dataset[i]])
            batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}

            n_surface = batch["surface_anchor_position"].shape[1]
            pred = ae_model.decode(
                mean_latent / latent_scale,
                batch["surface_anchor_position"],
                batch["volume_anchor_position"],
                batch.get("geometry_position"),
                batch.get("geometry_supernode_idx"),
            )

            sample_metrics = compute_field_metrics(pred, batch)
            for k, v in sample_metrics.items():
                all_metrics.setdefault(k, []).append(v)

            if (i + 1) % 5 == 0 or i == n - 1:
                print(f"  [{desc}] {i + 1}/{n}")

    avg = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    avg["total_mse"] = float(np.mean([np.mean(v) for v in all_metrics.values()]))
    return avg


def plot_metrics_comparison(
    ae_metrics: dict[str, float],
    diff_metrics: dict[str, float],
    title: str = "AE vs Diffusion per-field MSE",
):
    """bar chart comparing per-field MSE between AE reconstruction and diffusion generation."""
    fields = [k for k in ae_metrics if k.endswith("_mse") and k != "total_mse"]
    fields.sort()

    labels = [f.replace("_mse", "").replace("_", " ") for f in fields]
    ae_vals = [ae_metrics[f] for f in fields]
    diff_vals = [diff_metrics.get(f, 0) for f in fields]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, ae_vals, width, label="AE reconstruction", color="tab:blue", alpha=0.8)
    ax.bar(x + width / 2, diff_vals, width, label="Diffusion generation", color="tab:red", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    plt.show()

    # print table
    print(f"\n{'field':<25} {'AE MSE':>12} {'Diff MSE':>12} {'ratio':>8}")
    print("-" * 60)
    for f, label in zip(fields, labels, strict=True):
        ae_v = ae_metrics[f]
        diff_v = diff_metrics.get(f, float("nan"))
        ratio = diff_v / ae_v if ae_v > 0 else float("nan")
        print(f"{label:<25} {ae_v:>12.6f} {diff_v:>12.6f} {ratio:>8.2f}x")
    print("-" * 60)
    ae_total = ae_metrics.get("total_mse", 0)
    diff_total = diff_metrics.get("total_mse", 0)
    ratio = diff_total / ae_total if ae_total > 0 else float("nan")
    print(f"{'TOTAL':<25} {ae_total:>12.6f} {diff_total:>12.6f} {ratio:>8.2f}x")


def load_best_checkpoint(model, output_dir: str | Path, model_name: str, device: str = "cuda"):
    """find and load the best checkpoint for a model from an output directory."""
    output_dir = Path(output_dir)
    for run_dir in sorted(output_dir.iterdir(), reverse=True):
        ckpt = run_dir / "checkpoints" / f"{model_name}_cp=best_model.loss.test.total_model.th"
        if ckpt.exists():
            state = torch.load(ckpt, map_location=device, weights_only=True)
            model.load_state_dict(state["state_dict"])
            print(f"loaded: {ckpt}")
            return ckpt
    print(f"warning: no best checkpoint found in {output_dir}")
    return None


def load_latest_checkpoint(model, output_dir: str | Path, model_name: str, device: str = "cuda"):
    """find and load the latest checkpoint for a model."""
    output_dir = Path(output_dir)
    for run_dir in sorted(output_dir.iterdir(), reverse=True):
        ckpt = run_dir / "checkpoints" / f"{model_name}_cp=latest_model.th"
        if ckpt.exists():
            state = torch.load(ckpt, map_location=device, weights_only=True)
            model.load_state_dict(state["state_dict"])
            print(f"loaded: {ckpt}")
            return ckpt
    print(f"warning: no latest checkpoint found in {output_dir}")
    return None
