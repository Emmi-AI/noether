#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.data.schemas import FileMap

SURFACE_POINTS_FILE = "surface_points.pt"
SURFACE_PRESSURE_FILE = "surface_pressure.pt"
SURFACE_NORMALS_FILE = "surface_normals.pt"
VOLUME_VELOCITY_FILE = "volume_velocity.pt"
VOLUME_POINTS_FILE = "volume_points.pt"
VOLUME_SDF_FILE = "volume_sdf.pt"
VOLUME_NORMALS_FILE = "volume_normals.pt"

SHAPENET_CAR_FILEMAP = FileMap(
    surface_position=SURFACE_POINTS_FILE,
    surface_pressure=SURFACE_PRESSURE_FILE,
    volume_velocity=VOLUME_VELOCITY_FILE,
    volume_position=VOLUME_POINTS_FILE,
    volume_distance_to_surface=VOLUME_SDF_FILE,
    surface_normals=SURFACE_NORMALS_FILE,
    volume_normals=VOLUME_NORMALS_FILE,
)
