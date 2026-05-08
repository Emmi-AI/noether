#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Force coefficient computation: ground-truth and predicted Cd/Cl comparison."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from aero_cfd.utils.drag_lift import FlowConditions, compute_force_coefficients


def compute_forces_for_split(
    dataset_root: str,
    split: str,
    predictions_path: str,
) -> tuple[list[dict], Path]:
    """Compute ground-truth and predicted Cd/Cl for every sample in a split.

    For each sample, loads surface pressure, wall shear stress, normals, and
    cell areas from the dataset directory.  If a saved predictions file exists,
    also computes predicted Cd/Cl by matching prediction positions to the
    dataset mesh via nearest-neighbor lookup.

    Args:
        dataset_root: Path to the DrivAerML dataset.
        split: Dataset split name (``'test'``, ``'val'``, etc.).
        predictions_path: Directory containing ``sample_NNNN.pt`` files.

    Returns:
        Tuple of ``(rows, csv_path)`` where *rows* is a list of per-sample
        dicts and *csv_path* is the output CSV location.
    """
    from scipy.spatial import cKDTree

    from noether.data.datasets.cfd.caeml.drivaerml.split import DrivAerMLDefaultSplitIDs

    design_ids = sorted(DrivAerMLDefaultSplitIDs().model_dump()[split])
    pred_dir = Path(predictions_path)
    csv_path = pred_dir / "forces.csv"

    required_files = [
        "surface_pressure.pt",
        "surface_wallshearstress.pt",
        "surface_normal_vtp.pt",
        "surface_area_vtp.pt",
        "surface_position_vtp.pt",
    ]

    rows: list[dict] = []

    for idx, design_id in enumerate(design_ids):
        run_dir = Path(dataset_root) / f"run_{design_id}"

        missing = [f for f in required_files if not (run_dir / f).exists()]
        if missing:
            rows.append({"sample": idx, "run_id": design_id, "skipped": missing})
            continue

        # Ground-truth fields
        ground_truth_pressure = torch.load(
            run_dir / "surface_pressure.pt", map_location="cpu", weights_only=True
        ).float()
        ground_truth_shear_stress = torch.load(
            run_dir / "surface_wallshearstress.pt", map_location="cpu", weights_only=True
        ).float()
        surface_normals = torch.load(run_dir / "surface_normal_vtp.pt", map_location="cpu", weights_only=True).float()
        surface_areas = torch.load(run_dir / "surface_area_vtp.pt", map_location="cpu", weights_only=True).float()
        surface_positions = torch.load(
            run_dir / "surface_position_vtp.pt", map_location="cpu", weights_only=True
        ).float()

        if ground_truth_pressure.ndim == 2 and ground_truth_pressure.shape[-1] == 1:
            ground_truth_pressure = ground_truth_pressure.squeeze(-1)

        flow = _load_flow_conditions(run_dir, design_id)
        ground_truth_coefficients = compute_force_coefficients(
            ground_truth_pressure,
            ground_truth_shear_stress,
            surface_normals,
            surface_areas,
            flow,
        )

        row: dict = {
            "sample": idx,
            "run_id": design_id,
            "ref_area": flow.reference_area,
            "gt_cells": ground_truth_pressure.shape[0],
            "gt_cd": round(ground_truth_coefficients.cd, 6),
            "gt_cl": round(ground_truth_coefficients.cl, 6),
            "pred_cells": "",
            "pred_cd": "",
            "pred_cl": "",
            "drag_error": "",
            "lift_error": "",
        }

        # Predicted Cd/Cl (if predictions exist for this sample)
        prediction_file = pred_dir / f"sample_{idx:04d}.pt"
        if prediction_file.exists():
            saved_data = torch.load(prediction_file, map_location="cpu", weights_only=True)
            predicted_pressure = saved_data.get("surface_pressure")
            predicted_shear_stress = saved_data.get("surface_friction")
            predicted_positions = saved_data.get("surface_anchor_position")

            if (
                predicted_pressure is not None
                and predicted_shear_stress is not None
                and predicted_positions is not None
            ):
                if predicted_pressure.ndim == 2 and predicted_pressure.shape[-1] == 1:
                    predicted_pressure = predicted_pressure.squeeze(-1)

                position_tree = cKDTree(surface_positions.numpy())
                _, matched_indices = position_tree.query(predicted_positions.numpy())

                predicted_coefficients = compute_force_coefficients(
                    predicted_pressure,
                    predicted_shear_stress,
                    surface_normals[matched_indices],
                    surface_areas[matched_indices],
                    flow,
                )
                row.update(
                    {
                        "pred_cells": predicted_pressure.shape[0],
                        "pred_cd": round(predicted_coefficients.cd, 6),
                        "pred_cl": round(predicted_coefficients.cl, 6),
                        "drag_error": round(abs(ground_truth_coefficients.cd - predicted_coefficients.cd), 6),
                        "lift_error": round(abs(ground_truth_coefficients.cl - predicted_coefficients.cl), 6),
                    }
                )

        rows.append(row)

    # Write CSV
    completed_rows = [r for r in rows if "skipped" not in r]
    if completed_rows:
        pred_dir.mkdir(parents=True, exist_ok=True)
        fieldnames = [k for k in completed_rows[0] if k != "skipped"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(completed_rows)

    return rows, csv_path


def _load_flow_conditions(run_dir: Path, design_id: int) -> FlowConditions:
    """Load per-run reference area from CSV, falling back to defaults."""
    ref_csv = run_dir / f"geo_ref_{design_id}.csv"
    if ref_csv.exists():
        import pandas as pd

        ref_area = float(pd.read_csv(ref_csv)["aRef"][0])
        return FlowConditions(reference_area=ref_area)
    return FlowConditions()
