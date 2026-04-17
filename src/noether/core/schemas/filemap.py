#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from pydantic import BaseModel


class FileMap(BaseModel):
    """File mapping schema for aerodynamic datasets.

    Maps field names to their corresponding file names in the dataset directory.
    This allows different datasets to use different file naming conventions while maintaining a unified interface.
    """

    # Surface field files
    surface_position: str | None = None
    surface_pressure: str | None = None
    surface_friction: str | None = None
    surface_normals: str | None = None
    surface_area: str | None = None

    # Volume field files
    volume_position: str | None = None
    volume_pressure: str | None = None
    volume_velocity: str | None = None
    volume_vorticity: str | None = None
    volume_normals: str | None = None

    # Optional additional surface position files (dataset-specific)
    surface_position_stl: str | None = None
    surface_position_stl_resampled: str | None = None

    # Optional volume friction
    volume_friction: str | None = None

    # Optional volume distance field
    volume_distance_to_surface: str | None = None

    # Optional design parameters file
    design_parameters: str | None = None
