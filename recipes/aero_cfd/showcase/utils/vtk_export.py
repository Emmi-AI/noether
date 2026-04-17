#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from pathlib import Path

import torch


def is_pyvista_available() -> bool:
    """Check whether pyvista is importable."""
    try:
        import pyvista as pv  # noqa: F401

        return True
    except ImportError:
        return False


def export_pointcloud_to_vtk(
    positions: torch.Tensor,
    fields: dict[str, torch.Tensor],
    output_path: str,
) -> str:
    """Create a VTK point cloud from position and field tensors.

    Each entry in ``fields`` is added as point data on the cloud.
    Tensor shape determines the type: ``(N,)`` becomes scalar data,
    ``(N, D)`` becomes vector/multi-component data.

    Args:
        positions: Point positions, shape ``(N, 3)``.
        fields: Mapping from field name to value tensor.
        output_path: Path to write the output VTP file.

    Returns:
        The path to the written file.

    Raises:
        ImportError: If pyvista is not installed.
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("pyvista is required for VTK export. Install with: pip install pyvista") from None

    cloud = pv.PolyData(positions.cpu().numpy())

    for name, tensor in fields.items():
        data = tensor.cpu().numpy()
        # Flatten (N, 1) scalars to (N,)
        if data.ndim == 2 and data.shape[1] == 1:
            data = data.squeeze(1)
        cloud[name] = data

    cloud.save(output_path)
    return output_path


def export_all_samples(
    predictions_path: str,
    domain_position_keys: dict[str, str],
) -> Path:
    """Export all saved prediction samples as per-domain VTP point clouds.

    For each ``sample_NNNN.pt`` file, creates one VTP file per domain
    (surface, volume) in a ``vtk/`` subdirectory.

    Args:
        predictions_path: Directory containing ``sample_NNNN.pt`` files.
        domain_position_keys: Mapping from domain prefix to the position key
            in the saved dict (e.g. ``{"surface": "surface_anchor_position"}``).

    Returns:
        Path to the ``vtk/`` output directory.
    """
    pred_dir = Path(predictions_path)
    sample_files = sorted(pred_dir.glob("sample_*.pt"))
    vtk_dir = pred_dir / "vtk"
    vtk_dir.mkdir(parents=True, exist_ok=True)

    for sample_file in sample_files:
        data = torch.load(sample_file, map_location="cpu", weights_only=True)
        if not isinstance(data, dict):
            continue

        for domain, pos_key in domain_position_keys.items():
            positions = data.get(pos_key)
            if positions is None:
                continue
            fields = {k: v for k, v in data.items() if k.startswith(f"{domain}_") and k != pos_key}
            if not fields:
                continue
            out_path = str(vtk_dir / f"{sample_file.stem}_{domain}.vtp")
            export_pointcloud_to_vtk(positions=positions, fields=fields, output_path=out_path)

    return vtk_dir
