#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import Any

from examples.presets.aero_cfd import AeroCFDPreset
from noether.core.schemas.dataset import AeroDataSpecs


class EmmiWingPreset(AeroCFDPreset):
    """Preset for the EMMI Wing CFD dataset."""

    _dataset_kind = "noether.data.datasets.cfd.EmmiWingDataset"

    _stats: dict[str, list[float]] = {
        "raw_pos_min": [-17.5],
        "raw_pos_max": [17.5],
        "surface_pressure_mean": [92656.34610807039],
        "surface_pressure_std": [11929.058756240694],
        "surface_friction_mean": [-74.10092405045339, -0.5525946509854017, 0.0401677695420727],
        "surface_friction_std": [47.16838501471528, 10.233076648224564, 23.08224849769229],
        "volume_velocity_mean": [187.92724405048926, 0.5335961966484881, -0.0812512160659759],
        "volume_velocity_std": [83.6810800019851, 19.911990565773156, 33.3370080829507],
        "volume_pressure_mean": [93342.81261991762],
        "volume_pressure_std": [11743.515250769764],
        "volume_vorticity_logscale_mean": [-0.013138919851553849, 0.0033505699708222037, -1.6626923006758065],
        "volume_vorticity_logscale_std": [6.217698713293661, 8.787798800793741, 5.512463961862133],
        "volume_vorticity_magnitude_mean": [3.79027e04],
        "_zero": [0.0],
        "geometry_design_parameters_mean": [
            0.9495975525165955,
            1.2492765187383101,
            0.5500557258109463,
            19.92001709415523,
            0.0,
        ],
        "geometry_design_parameters_std": [
            0.14422534397647097,
            0.14465871522018967,
            0.08706049742886027,
            11.557458965186635,
            1.0,
        ],
        "inflow_design_parameters_mean": [224.59267997259747, -0.045877170231132774],
        "inflow_design_parameters_std": [43.25454930960363, 5.792067554535914],
    }

    _pipeline_defaults: dict[str, Any] = {
        "num_surface_points": 16384,
        "num_volume_points": 16384,
        "num_surface_queries": 0,
        "num_volume_queries": 0,
        "use_physics_features": False,
    }

    _pipeline_model_overrides: dict[str, dict[str, Any]] = {
        "noether.modeling.models.wrappers.UPTWrapper": {
            "num_supernodes": 16384,
            "sample_query_points": False,
            "num_surface_queries": 16384,
            "num_volume_queries": 16384,
        },
        "noether.modeling.models.wrappers.ABUPTWrapper": {
            "num_geometry_supernodes": 1024,
            "num_geometry_points": 16384,
            "num_surface_anchor_points": 512,
            "num_volume_anchor_points": 512,
            "num_surface_queries": 0,
            "num_volume_queries": 0,
        },
    }

    _forward_properties: dict[str, list[str]] = {
        "noether.modeling.models.wrappers.UPTWrapper": [
            "surface_mask_query",
            "surface_position_batch_idx",
            "surface_position_supernode_idx",
            "surface_position",
            "surface_query_position",
            "volume_query_position",
            "geometry_design_parameters",
            "inflow_design_parameters",
        ],
        "noether.modeling.models.wrappers.ABUPTWrapper": [
            "geometry_position",
            "geometry_supernode_idx",
            "geometry_batch_idx",
            "surface_anchor_position",
            "volume_anchor_position",
            "geometry_design_parameters",
            "inflow_design_parameters",
        ],
        "_default": [
            "surface_position",
            "volume_position",
            "surface_features",
            "volume_features",
            "geometry_design_parameters",
            "inflow_design_parameters",
        ],
    }

    @property
    def data_specs(self) -> AeroDataSpecs:
        return AeroDataSpecs(
            position_dim=3,
            surface_output_dims={"pressure": 1, "friction": 3},
            volume_output_dims={"pressure": 1, "velocity": 3, "vorticity": 3},
            conditioning_dims={
                "geometry_design_parameters": 5,
                "inflow_design_parameters": 2,
            },
        )

    @property
    def normalizer_spec(self) -> dict[str, str | tuple[str, dict[str, Any]]]:
        return {
            "surface_pressure": "mean_std",
            "surface_friction": "mean_std",
            "volume_pressure": "mean_std",
            "volume_velocity": "mean_std",
            # Wing uses magnitude-based normalization for vorticity (mean=0, std=magnitude_mean)
            "volume_vorticity": (
                "mean_std",
                {
                    "mean_key": "_zero",
                    "std_key": "volume_vorticity_magnitude_mean",
                },
            ),
            "geometry_design_parameters": "mean_std",
            "inflow_design_parameters": "mean_std",
            "surface_position": ("position", {"scale": 1000}),
            "volume_position": ("position", {"scale": 1000}),
        }

    @property
    def excluded_properties(self) -> set[str]:
        return {"surface_normals", "volume_normals", "volume_sdf"}

    def target_properties(self) -> list[str]:
        return [
            "surface_pressure_target",
            "surface_friction_target",
            "volume_pressure_target",
            "volume_velocity_target",
            "volume_vorticity_target",
        ]
