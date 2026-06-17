#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""
This script converts the vtk files from the DrivAerNet++ dataset to torch tensors.
It processes pressure and wall shear stress surface data, ensuring point order consistency,
and saves the processed data in a structured format.
Subsamples volume data for efficiency.

"""

import functools
import hashlib
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import click
import numpy as np
import pyvista as pv
import torch
import tqdm


def seed_from(*args):
    """
    hashes any arguments to create a unique seed. This is useful for reproducibility,
    e.g. when generating same data for different models (which increment the seed differently).
    -------------
    Example usage:
    with local_seed(seed_from("train", 10)):
        x = torch.randn(2)
        y = np.random.randn(2)
    z = torch.randn(2)  # here we have re-stored the "global" seed.
    """
    m = hashlib.sha256()
    for arg in args:
        m.update(str(arg).encode())
    h = int.from_bytes(m.digest(), "big")
    seed = h % (2**32 - 1)
    return seed


def get_vtk_files(root: Path) -> list[str]:
    vtk_files = []
    for sub_dir in os.listdir(root):
        for file in os.listdir(root / sub_dir):
            if file.endswith(".vtk"):
                vtk_files.append(f"{sub_dir}/{file}")
            else:
                print("Found non-vtk file: ", root / sub_dir / file)
    return vtk_files


def load_data(file_path: Path) -> dict[str, torch.Tensor]:
    """Loads position and field data from a VTK file into PyTorch tensors."""
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    mesh = pv.read(file_path)
    if "U" in mesh.array_names:  # hardcoded U is (volume) velocity.
        # Computing vorticity on cell level because it fails on point data due to non-invertible Jacobian.
        mesh = mesh.compute_derivative(scalars="U", gradient=False, vorticity=True, preference="cell")
        mesh = mesh.cell_data_to_point_data()

    blacklist = ["patchID", "cellID", "_PYVISTA_USER_DICT"]
    array_names = {name for name in mesh.array_names if name not in blacklist}
    positions = {"position": mesh.points}
    # Use .point_data instead of get_array(..., preference='point') because that is not strict (and we use mesh.points)
    fields = {field: mesh.point_data[field] for field in array_names}
    fields_and_positions = {**positions, **fields}
    return {field_name: torch.from_numpy(data.astype(np.float32)) for field_name, data in fields_and_positions.items()}


def reorder_data_via_position_map(
    reference_data: dict[str, torch.Tensor],
    data_to_reorder: dict[str, torch.Tensor],
    rounding_precision: int | None = None,
) -> tuple[dict[str, torch.Tensor], int]:
    """Reorders tensors in 'data_to_reorder' to match the point order of 'reference_data'.

    This function assumes that both datasets contain the exact same set of points, just in a different order.
    The mapping is done using a dictionary lookup where point coordinates are the keys.

    Args:
        reference_data: The dictionary with the target ordering (your pressure_data).
        data_to_reorder: The dictionary with the data to be reordered (your wallshear_data).

    Returns:
        A new dictionary containing the data from 'data_to_reorder' sorted
        according to the point order in 'reference_data'.
    """
    if rounding_precision:
        hash_point = lambda point: tuple(round(coord, rounding_precision) for coord in point.tolist())
    else:
        hash_point = lambda point: tuple(point.tolist())

    # 1. Create a lookup map from point coordinates to the original index in the data that needs reordering.
    positions_to_reorder = data_to_reorder["position"]
    point_to_original_index_map = {hash_point(point): i for i, point in enumerate(positions_to_reorder)}
    assert len(point_to_original_index_map) == len(positions_to_reorder), "Some points are not unique"

    # 2. Use the reference order to build a list of new indices.
    reference_positions = reference_data["position"]
    try:
        reordering_indices = [point_to_original_index_map[hash_point(point)] for point in reference_positions]
    except KeyError as e:
        print("Error: A point in the reference data was not found in the data to be reordered.")
        print(f"Missing point coordinate: {e.args[0]}")

        num_missing = 0
        for point in reference_positions:
            if hash_point(point) not in point_to_original_index_map:
                num_missing += 1

        print(f"Found {num_missing} missing points. Returning None.")
        return {}, num_missing
        # raise ValueError("Point sets do not seem to match.") from e

    reordering_indices_tensor = torch.tensor(reordering_indices, dtype=torch.long)

    # 3. Apply the reordering to all tensors in the dictionary.
    reordered_data = {
        field_name: tensor_data[reordering_indices_tensor] for field_name, tensor_data in data_to_reorder.items()
    }

    return reordered_data, 0


def process_single_case(
    vtk_file: str,
    pressure_root: Path,
    wallshear_root: Path,
    output_root: Path,
    volume_root: Path,
    subsample_factor_volume: int = 10,
):
    """
    Loads, reorders, and saves data for a single simulation case.

    Args:
        vtk_file: The relative path of the vtk file (e.g., 'experiment/data.vtk').
        pressure_root: The root directory for pressure data.
        wallshear_root: The root directory for wall shear stress data.
        output_root: The root directory to save the processed .pt files.
        volume_root: The root directory for volume data.
    """
    output_file_id = vtk_file.split("/")[-1].split(".")[0]
    folder_path = output_root / output_file_id
    os.makedirs(folder_path, exist_ok=True)

    # Surface data
    pressure_data = load_data(pressure_root / vtk_file)
    wallshear_data_unordered = load_data(wallshear_root / vtk_file)

    # For some reason they have their data shuffled in the two respective surface files (pressure, wallShearStress)...
    # Therefore, we now have to match the positions. Doing this via a hashmap where positions are the keys.
    wallshear_data, num_missing = reorder_data_via_position_map(
        reference_data=pressure_data,
        data_to_reorder=wallshear_data_unordered,
    )
    if num_missing == 0:  # almost all files expected to reach this
        assert torch.all(pressure_data["position"] == wallshear_data["position"]), "Positions do not match"
        for field_name, data in wallshear_data.items():
            assert len(data) == len(wallshear_data["position"]), f"Data length mismatch for {field_name}"
    else:  # for some reason a few wallshearstress files even have different mesh than pressure files...
        # add this simulation case to blacklist txt file and simply store the data without reordering (becase cannot)
        wallshear_data = wallshear_data_unordered
        with open(output_root / "blacklist.txt", "a") as f:
            f.write(f"{vtk_file}\n")

    surface_data = {**pressure_data, **wallshear_data}
    for field_name, data in surface_data.items():
        torch.save(data, folder_path / f"surface_{field_name}.pt")

    # Volume data
    prefix, suffix = vtk_file.split("/")
    part1_file = volume_root / f"{prefix}_part1" / suffix
    part2_file = volume_root / f"{prefix}_part2" / suffix
    assert part1_file.exists() ^ part2_file.exists(), f"Exactly one of {part1_file} and {part2_file} should exist."
    if part1_file.exists():
        volume_data = load_data(part1_file)
    elif part2_file.exists():
        volume_data = load_data(part2_file)
    else:
        raise FileNotFoundError(f"Neither {part1_file} nor {part2_file} exist.")
    for field_name, data in volume_data.items():
        assert len(data) == len(volume_data["position"]), f"Data length mismatch for {field_name}"
        seed = seed_from(vtk_file, "volume")
        perm = torch.randperm(len(volume_data["position"]), generator=torch.Generator().manual_seed(seed))
        perm = perm[: len(perm) // subsample_factor_volume]
        torch.save(data[perm], folder_path / f"volume_{field_name}.pt")


@click.command()
@click.option(
    "--unzipped-root",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path),
    required=True,
    help="Path to the root directory of the unzipped raw dataset (e.g., '/path/to/drivaernet/raw/unzipped').",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
    required=True,
    help="Path to the directory where preprocessed files will be saved (e.g., '/path/to/drivaernet/preprocessed').",
)
@click.option(
    "--subsample-factor-volume",
    type=int,
    default=10,
    show_default=True,
    help="Factor by which to subsample the volume data points.",
)
def main(unzipped_root: Path, output_dir: Path, subsample_factor_volume: int):
    """Main function to set up paths and process all files."""
    pressure_path = unzipped_root / "PressureVTK"
    wallshear_path = unzipped_root / "WallShearStressVTK_Updated"
    volume_path = unzipped_root / "CFD"

    experiment_folders = set(os.listdir(pressure_path))
    experiment_folders_volume = {f"{name}_part1" for name in experiment_folders} | {
        f"{name}_part2" for name in experiment_folders
    }
    assert os.path.exists(unzipped_root), f"Path {unzipped_root} does not exist"
    assert set(os.listdir(wallshear_path)) == experiment_folders, (
        f"folders in {wallshear_path} do not match {pressure_path}"
    )
    assert set(os.listdir(volume_path)) == experiment_folders_volume, (
        f"folders in {volume_path} are expected as {pressure_path} but part_1 and part_2 each."
    )

    vtk_files = get_vtk_files(pressure_path)
    wallshear_vtk_files = get_vtk_files(wallshear_path)
    out_experiment_names = [vtk_file.split("/")[-1].split(".")[0] for vtk_file in vtk_files]
    assert len(set(out_experiment_names)) == len(out_experiment_names), "Some experiment names are not unique"
    assert len(set(out_experiment_names)) == 8129, f"Expected 8129 files, but found {len(out_experiment_names)} files"
    assert set(vtk_files) == set(wallshear_vtk_files), "VTK files do not match"

    # if blacklist file exists, ask if should overwrite and do overwrite. Else create new
    blacklist_path = output_dir / "blacklist.txt"
    if blacklist_path.exists():
        if not click.confirm(f"Blacklist file already exists at {blacklist_path}. Overwrite?"):
            sys.exit(0)
        else:
            blacklist_path.unlink()
            print(f"Blacklist file {blacklist_path} deleted.")

    # Use a multiprocessing Pool to process files in parallel.
    # The number of processes will default to os.cpu_count().
    with Pool(processes=140) as pool:
        # Use map to apply process_single_case to each file.
        # Wrap with tqdm for a progress bar. list() consumes the iterator.
        list(
            tqdm.tqdm(
                pool.map(
                    functools.partial(
                        process_single_case,
                        pressure_root=pressure_path,
                        wallshear_root=wallshear_path,
                        output_root=output_dir,
                        volume_root=volume_path,
                        subsample_factor_volume=subsample_factor_volume,
                    ),
                    vtk_files,
                ),
                total=len(vtk_files),
                desc="Processing VTK files",
            )
        )


if __name__ == "__main__":
    """This script converts the vtk files from the DrivAerNet++ dataset to torch tensors."""
    main()
