#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from noether.core.schemas.statistics import AeroStatsSchema


@pytest.mark.parametrize(
    ("field_name", "valid_value"),
    [
        # Coordinate/Vector fields (3-tuples):
        ("surface_domain_min", (0.0, -1.0, 0.5)),
        ("surface_friction_std", (0.1, 0.1, 0.1)),
        ("volume_velocity_mean", (10.5, 0.0, -2.1)),
        # Scalar statistics (1-tuples or floats):
        ("surface_pressure_mean", (101325.0,)),
        ("volume_vorticity_magnitude_std", 0.05),
        ("volume_sdf_mean", (-0.5,)),
        # Sequences (design parameters):
        ("geometry_design_parameters_min", [0.1, 0.5, 0.9, 1.2]),
        ("inflow_design_parameters_mean", (25.0, 30.0)),
    ],
)
def test_aero_stats_field_types(field_name: str, valid_value: Sequence[float]) -> None:
    stats = AeroStatsSchema(**{field_name: valid_value})
    assert getattr(stats, field_name) == valid_value


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("surface_domain_min", (1.0, 2.0)),  # Too short
        ("volume_velocity_std", (1.0, 2.0, 3.0, 4.0)),  # Too long
        ("surface_pressure_mean", 101325.0),  # Should be a tuple (101325.0,)
        ("volume_vorticity_magnitude_mean", "high"),  # Non-numeric
    ],
)
def test_aero_stats_invalid_shapes(field_name: str, invalid_value: Sequence[float]) -> None:
    with pytest.raises(ValidationError):
        AeroStatsSchema(**{field_name: invalid_value})


def test_aero_stats_full_load() -> None:
    data = {
        "surface_domain_min": (-1.0, -0.5, 0.0),
        "surface_domain_max": (1.0, 0.5, 2.0),
        "surface_pressure_mean": (0.0,),
        "volume_velocity_mean": (30.0, 0.0, 0.0),
        "volume_vorticity_magnitude_mean": 1.2,
        "geometry_design_parameters_mean": [0.5, 0.2, 0.1],
    }

    stats = AeroStatsSchema(**data)

    # Ensure volume and surface don't overwrite each other:
    assert stats.surface_domain_min != stats.volume_domain_min
    assert stats.volume_vorticity_magnitude_mean == 1.2

    exported = stats.model_dump(exclude_none=True)
    assert len(exported) == 6
    assert exported["surface_domain_min"] == (-1.0, -0.5, 0.0)
