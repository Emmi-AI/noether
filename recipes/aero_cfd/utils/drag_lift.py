#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class FlowConditions:
    """Simulation-specific constants for aerodynamic force computation.

    Default values correspond to DrivAerML dataset conditions.
    """

    rho: float = 1.0
    """Fluid density [kg/m^3]."""
    v_inf: float = 38.889
    """Freestream velocity magnitude [m/s]."""
    reference_area: float = 1.079
    """Reference area for coefficient normalization [m^2]."""
    inlet_direction: torch.Tensor = field(default_factory=lambda: torch.tensor([1.0, 0.0, 0.0]))
    """Unit vector in the drag (freestream) direction."""
    lift_direction: torch.Tensor = field(default_factory=lambda: torch.tensor([0.0, 0.0, 1.0]))
    """Unit vector in the lift direction."""


@dataclass
class ForceCoefficients:
    """Computed aerodynamic force coefficients."""

    cd: float
    """Drag coefficient."""
    cl: float
    """Lift coefficient."""


def compute_force_coefficients(
    surface_pressure: torch.Tensor,
    wall_shear_stress: torch.Tensor,
    surface_normals: torch.Tensor,
    surface_areas: torch.Tensor,
    flow_conditions: FlowConditions,
) -> ForceCoefficients:
    """Compute drag and lift coefficients from surface predictions and mesh geometry.

    Integrates pressure and shear forces over the surface mesh to obtain total aerodynamic
    force, then projects onto inlet/lift directions and normalizes by dynamic pressure.

    Args:
        surface_pressure: Predicted surface pressure values, shape ``(N,)``.
        wall_shear_stress: Predicted wall shear stress (friction) vectors, shape ``(N, 3)``.
        surface_normals: Outward-facing surface normal vectors from mesh, shape ``(N, 3)``.
        surface_areas: Cell areas from mesh, shape ``(N,)``.
        flow_conditions: Physical constants for the simulation.

    Returns:
        Drag and lift coefficients.
    """
    # Pressure force: F_p = sum(n * p * dA)
    pressure_force = (surface_normals.T * (surface_pressure * surface_areas)).sum(dim=1)

    # Shear force: F_s = -sum(tau * dA)
    shear_force = -(wall_shear_stress.T * surface_areas).sum(dim=1)

    total_force = pressure_force + shear_force

    # Dynamic pressure * reference area
    q_ref = 0.5 * flow_conditions.rho * flow_conditions.v_inf**2 * flow_conditions.reference_area

    cd = torch.dot(total_force, flow_conditions.inlet_direction) / q_ref
    cl = torch.dot(total_force, flow_conditions.lift_direction) / q_ref

    return ForceCoefficients(cd=cd.item(), cl=cl.item())


def load_mesh_geometry(vtp_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load surface normals and cell areas from a VTP mesh file.

    Args:
        vtp_path: Path to a VTP (VTK PolyData) mesh file.

    Returns:
        Tuple of ``(normals, areas)`` where normals has shape ``(N, 3)``
        and areas has shape ``(N,)``.

    Raises:
        ImportError: If pyvista is not installed.
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("pyvista is required to load mesh geometry. Install with: uv pip install pyvista") from None

    mesh = pv.read(vtp_path)
    normals = torch.from_numpy(mesh.cell_normals).float()
    areas = torch.from_numpy(mesh.compute_cell_sizes(length=False, volume=False)["Area"]).float()
    return normals, areas
