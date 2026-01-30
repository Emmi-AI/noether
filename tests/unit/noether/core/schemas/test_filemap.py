#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import pytest

from noether.core.schemas.filemap import FileMap


@pytest.mark.parametrize(
    ("field_name", "filename"),
    [
        ("surface_position", "wing_nodes.pt"),
        ("surface_position_stl", "airfoil_mesh.stl"),
        ("volume_velocity", "flow_field_001.vtu"),
        ("volume_pressure", "pressure_distribution.npz"),
        ("design_parameters", "config_v2.json"),
        ("volume_distance_to_surface", "dist_map.pt"),
    ],
)
def test_file_map_field_assignments(field_name: str, filename: str) -> None:
    f_map = FileMap(**{field_name: filename})

    assert getattr(f_map, field_name) == filename

    if field_name != "surface_friction":
        assert f_map.surface_friction is None


def test_file_map_from_dict() -> None:
    config_data = {
        "surface_position": "surf_pos.stl",
        "volume_pressure": "vol_p.vtu",
        "design_parameters": "params.json",
    }
    f_map = FileMap(**config_data)

    # model_dump(exclude_none=True) is great for cleaning up exports
    exported = f_map.model_dump(exclude_none=True)
    assert exported == config_data
