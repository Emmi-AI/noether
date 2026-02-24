#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.schemas.filemap import FileMap

CAEML_FILEMAP = FileMap(
    surface_position="surface_position_vtp.pt",
    surface_pressure="surface_pressure.pt",
    surface_friction="surface_wallshearstress.pt",
    surface_normals="surface_normal_vtp.pt",
    volume_position="volume_cell_position.pt",
    volume_pressure="volume_cell_totalpcoeff.pt",
    volume_velocity="volume_cell_velocity.pt",
    volume_vorticity="volume_cell_vorticity.pt",
)
