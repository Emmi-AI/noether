#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

"""
Preprocessing script for ShapeNet car CFD dataset.

This script processes raw VTK files containing CFD simulation data and extracts
surface and volume data for machine learning applications.
"""

import logging
import os
import sys
from pathlib import Path

import meshio  # type: ignore[import-untyped,import-not-found]
import numpy as np
import numpy.typing as npt
import torch
import vtk  # type: ignore[import-untyped]
from sklearn.neighbors import NearestNeighbors  # type: ignore[import-untyped]
from tqdm import tqdm
from vtk.util.numpy_support import vtk_to_numpy  # type: ignore[import-untyped]

from noether.data.datasets.cfd.shapenet_car.filemap import SHAPENET_CAR_FILEMAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_SIMULATION_COUNT = 889
NUM_PARAM_FOLDERS = 9
EPSILON = 1e-8
PRESSURE_SURFACE_FILE = "quadpress_smpl.vtk"
VELOCITY_VOLUME_FILE = "hexvelo_smpl.vtk"
PRESSURE_ARRAY_FILE = "press.npy"
EXPECTED_CELL_TYPE = "quad"

# Simulations excluded due to missing required files
EXCLUDED_SIMULATIONS = frozenset(
    [
        "854bb96a96a4d1b338acbabdc1252e2f",
        "85bb9748c3836e566f81b21e2305c824",
        "9ec13da6190ab1a3dd141480e2c154d3",
        "c5079a5b8d59220bc3fb0d224baae2a",
    ]
)


def load_unstructured_grid_data(file_path: Path) -> vtk.vtkUnstructuredGrid:
    """
    Load unstructured grid data from a VTK file.

    Args:
        file_path: Path to the VTK file containing unstructured grid data

    Returns:
        vtkUnstructuredGrid object containing the loaded data
    """
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(str(file_path))
    reader.Update()
    output = reader.GetOutput()
    return output


def unstructured_grid_data_to_poly_data(
    unstructured_grid_data: vtk.vtkUnstructuredGrid,
) -> vtk.vtkPolyData:
    """
    Convert unstructured grid data to poly data by extracting the surface.

    Args:
        unstructured_grid_data: vtkUnstructuredGrid object to convert

    Returns:
        vtkPolyData object representing the surface

    Note:
        The surface_filter is kept alive to maintain VTK object references.
    """
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(unstructured_grid_data)
    surface_filter.Update()
    poly_data = surface_filter.GetOutput()
    return poly_data


def get_sdf(
    target_points: npt.NDArray[np.float64], boundary_points: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Calculate signed distance field and normal directions from target points to boundary.

    Args:
        target_points: (N, 3) array of points where SDF is computed
        boundary_points: (M, 3) array of boundary surface points

    Returns:
        Tuple of (distances, directions) where:
        - distances: (N,) array of distances to nearest boundary point
        - directions: (N, 3) array of normalized direction vectors to nearest boundary point

    """
    nearest_neighbors = NearestNeighbors(n_neighbors=1).fit(boundary_points)
    distances, indices = nearest_neighbors.kneighbors(target_points)
    nearest_boundary_points = np.array([boundary_points[i[0]] for i in indices])
    directions = (target_points - nearest_boundary_points) / (distances + EPSILON)
    return distances.reshape(-1), directions


def get_normal(unstructured_grid_data: vtk.vtkUnstructuredGrid):
    """
    Compute normalized surface normals from unstructured grid data.

    Args:
        unstructured_grid_data: vtkUnstructuredGrid object containing the mesh

    Returns:
        (N, 3) array of normalized normal vectors at each point

    Raises:
        RuntimeError: If NaN values are detected in the computed normals

    Note:
        This function modifies the input unstructured_grid_data by setting cell normals.
    """
    poly_data = unstructured_grid_data_to_poly_data(unstructured_grid_data)
    normal_filter = vtk.vtkPolyDataNormals()
    normal_filter.SetInputData(poly_data)
    normal_filter.SetAutoOrientNormals(1)
    normal_filter.SetConsistency(1)
    normal_filter.SetComputeCellNormals(1)
    normal_filter.SetComputePointNormals(0)
    normal_filter.Update()

    unstructured_grid_data.GetCellData().SetNormals(normal_filter.GetOutput().GetCellData().GetNormals())
    cell_to_point = vtk.vtkCellDataToPointData()
    cell_to_point.SetInputData(unstructured_grid_data)
    cell_to_point.Update()
    normals = vtk_to_numpy(cell_to_point.GetOutput().GetPointData().GetNormals()).astype(np.double)
    normals /= np.max(np.abs(normals), axis=1, keepdims=True) + EPSILON
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + EPSILON
    if np.isnan(normals).sum() > 0:
        raise RuntimeError("NaN values detected in computed surface normals")
    return normals


def get_simulation_relative_paths(root: Path) -> list[Path]:
    """
    Get relative paths to all valid simulation directories in the dataset.

    The dataset is organized into 9 parameter folders (param0-param8), each containing
    multiple simulation subdirectories. Some simulations are excluded due to missing required files.

    Args:
        root: Path to the root directory containing param folders

    Returns:
        List of Path objects representing relative paths to valid simulations

    Raises:
        ValueError: If the expected number of simulations is not found
        FileNotFoundError: If a parameter folder does not exist
    """
    if not root.is_absolute():
        raise ValueError(f"Root path must be absolute: {root}")

    simulation_relative_paths = []
    for param_idx in range(NUM_PARAM_FOLDERS):
        param_folder = f"param{param_idx}"
        param_path = root / param_folder

        if not param_path.exists():
            raise FileNotFoundError(f"Parameter folder does not exist: {param_path}")

        for simulation_name in sorted(os.listdir(param_path)):
            if simulation_name in EXCLUDED_SIMULATIONS:
                logger.debug(f"Skipping excluded simulation: {simulation_name}")
                continue

            simulation_path = param_path / simulation_name
            if simulation_path.is_dir():
                simulation_relative_paths.append(simulation_path.relative_to(root))

    # 100 + 99 + 97 + 100 + 100 + 96 + 100 + 98 + 99 = 889 samples
    if len(simulation_relative_paths) != EXPECTED_SIMULATION_COUNT:
        raise ValueError(f"Expected {EXPECTED_SIMULATION_COUNT} simulations but found {len(simulation_relative_paths)}")
    return simulation_relative_paths


def load_simulation_data(
    root: Path, simulation_relative_path: Path
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Load and process data for a single simulation.

    Args:
        root: Root directory containing raw data
        simulation_relative_path: Relative path to the simulation directory

    Returns:
        Tuple of (surface_pressure, surface_position, surface_normals, mask,
                 exterior_points, exterior_velocity, exterior_sdf, exterior_normals)

    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If data validation fails
    """
    pressure_file_path = root / simulation_relative_path / PRESSURE_SURFACE_FILE
    velocity_file_path = root / simulation_relative_path / VELOCITY_VOLUME_FILE

    if not pressure_file_path.exists():
        raise FileNotFoundError(f"Pressure file does not exist: {pressure_file_path}")
    if not velocity_file_path.exists():
        raise FileNotFoundError(f"Velocity file does not exist: {velocity_file_path}")

    # Load grid data
    pressure_grid_data = load_unstructured_grid_data(pressure_file_path)
    velocity_grid_data = load_unstructured_grid_data(velocity_file_path)

    surface_pressure = vtk_to_numpy(pressure_grid_data.GetPointData().GetScalars())
    surface_position = vtk_to_numpy(pressure_grid_data.GetPoints().GetData())

    # Surface .vtk file contains points that don't belong to the mesh -> create mask to filter them out
    mesh = meshio.read(pressure_file_path)
    if len(mesh.cells) != 1:
        raise ValueError(f"Expected 1 cell block but found {len(mesh.cells)} in {pressure_file_path}")
    cell_block = mesh.cells[0]
    if cell_block.type != EXPECTED_CELL_TYPE:
        raise ValueError(
            f"Expected cell type '{EXPECTED_CELL_TYPE}' but found '{cell_block.type}' in {pressure_file_path}"
        )
    unique_indices = np.unique(cell_block.data)
    mask = np.isin(np.arange(len(mesh.points)), unique_indices)

    # Validate pressure data consistency
    mesh_points = mesh.points
    pressure = np.load(root / simulation_relative_path / PRESSURE_ARRAY_FILE)
    if not np.allclose(surface_pressure, pressure):
        max_diff = np.max(np.abs(surface_pressure - pressure))
        raise ValueError(f"Surface pressure mismatch in {simulation_relative_path}, max difference: {max_diff}")
    if not np.allclose(surface_position, mesh_points):
        raise ValueError(f"Surface position mismatch in {simulation_relative_path}")

    # Load velocity and volume data
    velocity = vtk_to_numpy(velocity_grid_data.GetPointData().GetVectors())
    volume_points = vtk_to_numpy(velocity_grid_data.GetPoints().GetData())
    volume_sdf, volume_normals = get_sdf(volume_points, surface_position)
    surface_normals = get_normal(pressure_grid_data)

    if velocity.shape[0] != volume_points.shape[0]:
        raise ValueError(
            f"Shape mismatch in {simulation_relative_path}: velocity.shape[0]={velocity.shape[0]}, "
            f"volume_points.shape[0]={volume_points.shape[0]}"
        )

    # Identify points that are exterior to the surface (volume points)
    surface_point_set = {tuple(p) for p in surface_position}
    exterior_indices = [i for i, p in enumerate(volume_points) if tuple(p) not in surface_point_set]

    exterior_points = volume_points[exterior_indices]
    exterior_sdf = volume_sdf[exterior_indices]
    exterior_normals = volume_normals[exterior_indices]
    exterior_velocity = velocity[exterior_indices]

    # Validate surface data dimensions match
    if not (surface_position.shape[0] == surface_pressure.shape[0] == surface_normals.shape[0]):
        raise ValueError(
            f"Surface data shape mismatch in {simulation_relative_path}: "
            f"position={surface_position.shape[0]}, pressure={surface_pressure.shape[0]}, "
            f"normals={surface_normals.shape[0]}"
        )
    # Validate exterior data dimensions match
    if not (
        exterior_points.shape[0] == exterior_sdf.shape[0] == exterior_velocity.shape[0] == exterior_normals.shape[0]
    ):
        raise ValueError(
            f"Exterior data shape mismatch in {simulation_relative_path}: "
            f"points={exterior_points.shape[0]}, sdf={exterior_sdf.shape[0]}, "
            f"velocity={exterior_velocity.shape[0]}, normals={exterior_normals.shape[0]}"
        )

    return (
        surface_pressure,
        surface_position,
        surface_normals,
        mask,
        exterior_points,
        exterior_velocity,
        exterior_sdf,
        exterior_normals,
    )


def save_preprocessed_data(
    save_path: Path,
    surface_pressure: npt.NDArray,
    surface_position: npt.NDArray,
    surface_normals: npt.NDArray,
    mask: npt.NDArray,
    exterior_points: npt.NDArray,
    exterior_velocity: npt.NDArray,
    exterior_sdf: npt.NDArray,
    exterior_normals: npt.NDArray,
) -> None:
    """
    Save preprocessed simulation data to disk.

    Args:
        save_path: Directory where preprocessed data will be saved
        surface_pressure: Surface pressure values
        surface_position: Surface point positions
        surface_normals: Surface normal vectors
        mask: Boolean mask for valid surface points
        exterior_points: Exterior (volume) point positions
        exterior_velocity: Velocity at exterior points
        exterior_sdf: Signed distance field at exterior points
        exterior_normals: Normal vectors at exterior points
    """
    save_path.mkdir(parents=True, exist_ok=True)
    # The types are ignored because the filemap is defined manually, and it's expected the values to be valid strings:
    torch.save(torch.Tensor(surface_pressure[mask]), save_path / SHAPENET_CAR_FILEMAP.surface_pressure)  # type: ignore[operator]
    torch.save(torch.Tensor(surface_position[mask]), save_path / SHAPENET_CAR_FILEMAP.surface_position)  # type: ignore[operator]
    torch.save(torch.Tensor(surface_normals[mask]), save_path / SHAPENET_CAR_FILEMAP.surface_normals)  # type: ignore[operator]
    torch.save(torch.Tensor(exterior_velocity), save_path / SHAPENET_CAR_FILEMAP.volume_velocity)  # type: ignore[operator]
    torch.save(torch.Tensor(exterior_points), save_path / SHAPENET_CAR_FILEMAP.volume_position)  # type: ignore[operator]
    torch.save(torch.Tensor(exterior_normals), save_path / SHAPENET_CAR_FILEMAP.volume_normals)  # type: ignore[operator]
    torch.save(torch.Tensor(exterior_sdf), save_path / SHAPENET_CAR_FILEMAP.volume_distance_to_surface)  # type: ignore[operator]


def process_single_simulation(root: Path, simulation_relative_path: Path, output_dir: Path) -> None:
    """
    Process a single simulation: load data, validate, and save.

    Args:
        root: Root directory containing raw data
        simulation_relative_path: Relative path to the simulation
        output_dir: Output directory for preprocessed data

    Raises:
        Various exceptions from data loading and validation
    """
    (
        surface_pressure,
        surface_position,
        surface_normals,
        mask,
        exterior_points,
        exterior_velocity,
        exterior_sdf,
        exterior_normals,
    ) = load_simulation_data(root, simulation_relative_path)

    save_path = (
        output_dir / "preprocessed" / simulation_relative_path
    )  # always put the preprocessed data in "preprocessed" subfolder
    save_preprocessed_data(
        save_path,
        surface_pressure,
        surface_position,
        surface_normals,
        mask,
        exterior_points,
        exterior_velocity,
        exterior_sdf,
        exterior_normals,
    )


def main(
    root: Path,
    output_dir: Path,
    continue_on_error: bool = False,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, int]:
    """
    Preprocess ShapeNet car dataset.

    Args:
        root: Path to the root directory containing the raw data
        output_dir: Path to the output directory where preprocessed data will be saved
        continue_on_error: If True, continue processing on errors instead of stopping
        dry_run: If True, only validate data without saving
        overwrite: If True, allow overwriting existing output directory

    Returns:
        Dictionary with statistics: {
            'total': total simulations,
            'success': successfully processed,
            'failed': failed to process
        }

    Raises:
        FileNotFoundError: If root directory does not exist
        FileExistsError: If output directory exists and overwrite is False
    """
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")
    if output_dir.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    logger.info(f"Source: {root.as_posix()}")
    logger.info(f"Destination: {output_dir.as_posix()}")
    logger.info(f"Dry run: {dry_run}")
    logger.info(f"Continue on error: {continue_on_error}")

    simulation_relative_paths = get_simulation_relative_paths(root)
    logger.info(f"Found {len(simulation_relative_paths)} simulations")

    stats = {"total": len(simulation_relative_paths), "success": 0, "failed": 0}
    failed_simulations = []

    # Preprocess each simulation
    for simulation_relative_path in tqdm(simulation_relative_paths, desc="Processing simulations", unit="sim"):
        try:
            if dry_run:
                logger.debug(f"[DRY RUN] Would process: {simulation_relative_path}")
                # Still validate by loading data
                _ = load_simulation_data(root, simulation_relative_path)
            else:
                process_single_simulation(root, simulation_relative_path, output_dir)
            stats["success"] += 1
            logger.debug(f"Successfully processed: {simulation_relative_path}")
        except Exception as e:
            stats["failed"] += 1
            failed_simulations.append((simulation_relative_path, str(e)))
            logger.error(f"Failed to process {simulation_relative_path}: {e}")
            if not continue_on_error:
                logger.error("Stopping due to error. Use --continue-on-error to skip failed simulations.")
                raise

    # Summary
    logger.info("=" * 60)
    logger.info("Processing Summary:")
    logger.info(f"  Total simulations: {stats['total']}")
    logger.info(f"  Successfully processed: {stats['success']}")
    logger.info(f"  Failed: {stats['failed']}")

    if failed_simulations:
        logger.warning(f"\nFailed simulations ({len(failed_simulations)}):")
        for sim_path, error in failed_simulations:
            logger.warning(f"  - {sim_path}: {error}")

    logger.info("=" * 60)

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess ShapeNet car CFD dataset for machine learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=str,
        help="Path to the root directory containing the raw data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output directory where preprocessed data will be saved",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining simulations if one fails",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data without actually saving preprocessed files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output directory",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )
    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    root = Path(args.root)
    output_dir = Path(args.output_dir)

    try:
        stats = main(
            root=root,
            output_dir=output_dir,
            continue_on_error=args.continue_on_error,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )

        # Exit with appropriate code
        if stats["failed"] > 0:
            logger.warning(f"Completed with {stats['failed']} failures")
            sys.exit(1)
        else:
            logger.info("All simulations processed successfully")
            sys.exit(0)
    except KeyboardInterrupt:
        logger.error("Interrupted by user")
        sys.exit(130)
    except Exception:
        logger.exception("Fatal error occurred")
        sys.exit(1)
