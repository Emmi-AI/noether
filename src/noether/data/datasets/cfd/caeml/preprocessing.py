#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

"""Converts raw DrivAerML or AhmedML dataset into subsampled pytorch files"""

import argparse
import os
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from time import perf_counter

import numpy as np
import pyvista as pv
import torch
import trimesh
from scipy.fftpack import dst  # type: ignore[import-untyped]
from scipy.spatial import cKDTree  # type: ignore[import-untyped]
from tqdm import tqdm


@dataclass(frozen=True)
class DataFieldKeyMap:
    stl_prefix: str
    vtp_pressure: str
    vtp_wallshearstress: str
    vtu_pressure: str
    vtu_velocity: str
    vtu_totalp: str
    vtu_vorticity: str | None  # None means no vorticity field in the vtu file -> Compute it from velocity field


dataset_keymaps = {
    "drivaerml": DataFieldKeyMap(
        stl_prefix="drivaer",
        vtp_pressure="pMeanTrim",
        vtp_wallshearstress="wallShearStressMeanTrim",
        vtu_pressure="pMeanTrim",
        vtu_velocity="UMeanTrim",
        vtu_totalp="CptMeanTrim",
        vtu_vorticity=None,
    ),
    "ahmedml": DataFieldKeyMap(
        stl_prefix="ahmed",
        vtp_pressure="pMean",
        vtp_wallshearstress="wallShearStressMean",
        vtu_pressure="pMean",
        vtu_velocity="UMean",
        vtu_totalp="total(p)_coeffMean",
        vtu_vorticity="vorticityMean",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser("Data preprocessing for the DrivAerML and AhmedML dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["drivaerml", "ahmedml"],
        help="Name of the dataset ('drivaerml' or 'ahmedml').",
    )
    parser.add_argument(
        "--root",
        type=str,
        help="Root location of the raw data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/nfs-gpu/research/datasets/ahmedml_full/preprocessed_10x",
        help="Where to store the preprocessed data, e.g., ",
    )
    parser.add_argument(
        "--subsample_factor",
        type=int,
        default=10,
        help="All data is subsampled with this factor",
    )
    parser.add_argument(
        "--compute_surface_distance_via_trimesh",
        action="store_true",
        help="If this flag is set, the distance will be calculated via trimesh, otherwise approximated via cKDTree."
        "Computing the SDF via trimesh is more accurate as it interpolates on the mesh, "
        "but extremely expensive (4.6s per 1000 points) compared to cKDTree (< 0.01s per 1000 points)."
        "The differences between the two methods is on the order of millimeters.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers to use for processing. Default is 1 (no multiprocessing).",
    )
    return vars(parser.parse_args())


def assert_same_length(*arrays: np.ndarray | torch.Tensor) -> None:
    """Checks that all arrays have the same length.

    Args:
        *arrays: any number of arrays.
    """
    lengths = [len(arr) for arr in arrays]
    assert len(set(lengths)) == 1, f"Not all arrays have same length. Got lengths {lengths}"
    assert lengths[0] > 0, f"Arrays have length {lengths[0]}. Expected to be > 0."


@contextmanager
def time_block(run_folder, label: str):
    """A context manager to measure execution time of a code block.

    Args:
        run_folder: The name of the run folder. Will be printed.
        label: The name of the process/algorithm/code executed within the with block. Will be printed.

    Usage:
        with time_block("some task"):
            # code
    """
    print(f"({run_folder}) Starting '{label}'...")
    start = perf_counter()
    yield
    end = perf_counter()
    elapsed = end - start
    print(f"({run_folder}) Finished '{label}'; took {elapsed:.6f} seconds.")


def merge_vtu_parts(folder: str | Path):
    """Merge .part VTU fragments into a full .vtu file in the specified folder.

    This function is required for DrivAerML on Huggingface, where the volume data is split into two .part files.
    See files in https://huggingface.co/datasets/neashton/drivaerml/tree/main/run_1.

    Args:
        folder (str | Path): Path to the run_i directory containing .part files.

    Raises:
        NotADirectoryError: If the provided folder path is not a directory.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise NotADirectoryError(f"'{folder}' is not a valid directory.")

    # Find all *.part files and sort them
    parts = sorted(folder_path.glob("*.part"))
    if len(parts) == 0:
        print(f"Skipping {folder_path.name}: found no .part files")
        return

    # Derive output filename, e.g. "volume_237.vtu" from "volume_237.vtu.00.part"
    output_file = folder_path / parts[0].stem.rsplit(".", 1)[0]

    # Write merged VTU
    with output_file.open("wb") as out_f:
        for part in parts:
            out_f.write(part.read_bytes())

    # Remove fragment files
    for part in parts:
        part.unlink()

    print(f"Merged → {output_file}")


def process_run(
    run_folder: str,
    root: Path,
    output_dir: Path,
    data_field_keymap: DataFieldKeyMap,
    subsample_factor: int,
    compute_surface_distance_via_trimesh: bool,
) -> None:
    print(f"\n\nProcessing run {run_folder}... \n\n")
    # Create destination path
    run_id = int(run_folder.split("_")[1])
    run_dst_path = dst / f"run_{run_id}"
    run_dst_path.mkdir()

    # --- Merge VTP fragments ---
    merge_vtu_parts(root / run_folder)

    # --- Process STL file ---
    stl = trimesh.load(root / run_folder / f"{data_field_keymap.stl_prefix}_{run_id}.stl", force="mesh")
    torch.save(torch.Tensor(stl.vertices), run_dst_path / "surface_position_stl.pt")  # type: ignore[attr-defined]
    stl_resampled_pos, stl_resampled_faces = trimesh.sample.sample_surface_even(stl, 100_000)
    torch.save(torch.Tensor(stl_resampled_pos), run_dst_path / "surface_position_stl_resampled100k.pt")

    # Process face normals and resampled normals
    face_normals = stl.face_normals  # type: ignore[attr-defined]
    face_normals = face_normals / np.linalg.norm(face_normals, axis=1)[:, None]
    face_normals_resampled = stl.face_normals[stl_resampled_faces]  # type: ignore[attr-defined]
    face_normals_resampled = face_normals_resampled / np.linalg.norm(face_normals_resampled, axis=1)[:, None]

    print(f"({run_folder}) Storing stl file data")
    torch.save(torch.Tensor(face_normals), run_dst_path / "surface_normal_stl.pt")
    torch.save(torch.Tensor(face_normals_resampled), run_dst_path / "surface_normal_stl_resampled100k.pt")

    # --- Process VTP file ---
    with time_block(run_folder, "Load vtp file"):
        vtp = pv.read(root / run_folder / f"boundary_{run_id}.vtp")
    vtp_pressure = vtp.get_array(data_field_keymap.vtp_pressure)
    vtp_position = vtp.cell_centers().points
    vtp_wallshearstress = vtp.get_array(data_field_keymap.vtp_wallshearstress).astype(np.float32)
    # Per-cell surface areas, aligned with the cell centers used as positions.
    vtp_cell_area = vtp.compute_cell_sizes(length=False, area=True, volume=False).cell_data["Area"]
    assert_same_length(vtp_position, vtp_pressure, vtp_wallshearstress, vtp_cell_area)

    # Downsampling with a seeded permutation
    vtp_perm = torch.randperm(len(vtp_position), generator=torch.Generator().manual_seed(run_id))
    vtp_perm = vtp_perm[: len(vtp_perm) // subsample_factor]

    print(f"({run_folder}) Storing surface data")
    torch.save(torch.Tensor(vtp_position)[vtp_perm], run_dst_path / "surface_position_vtp.pt")
    torch.save(torch.Tensor(vtp_pressure)[vtp_perm], run_dst_path / "surface_pressure.pt")
    torch.save(torch.Tensor(vtp_wallshearstress)[vtp_perm], run_dst_path / "surface_wallshearstress.pt")
    torch.save(torch.Tensor(vtp_cell_area)[vtp_perm], run_dst_path / "surface_area_vtp.pt")

    # --- Process VTU file (Volume Data) ---
    with time_block(run_folder, "Load vtu file"):
        vtu = pv.read(root / run_folder / f"volume_{run_id}.vtu")
    vtu_cell_position = vtu.cell_centers().points
    vtu_cell_pressure = vtu.cell_data[data_field_keymap.vtu_pressure]
    vtu_cell_velocity = vtu.cell_data[data_field_keymap.vtu_velocity]
    vtu_cell_pressure_total = vtu.cell_data[data_field_keymap.vtu_totalp]

    # For vorticity, either compute it or load directly.
    if data_field_keymap.vtu_vorticity is not None:
        print(f"({run_folder}) Loading vorticity from vtu file")
        vtu_cell_vorticity = vtu.cell_data[data_field_keymap.vtu_vorticity]
    else:
        with time_block(run_folder, "Calculating vorticity (this can take a long time)"):
            vtu_deriv = vtu.compute_derivative(
                scalars=data_field_keymap.vtu_velocity,
                gradient=False,
                vorticity=True,
                preference="cell",
                progress_bar=True,
            )
            vtu_cell_vorticity = vtu_deriv.cell_data["vorticity"]

    assert_same_length(
        vtu_cell_position,
        vtu_cell_pressure,
        vtu_cell_velocity,
        vtu_cell_vorticity,
        vtu_cell_pressure,
    )

    # Create permutation for downsampling
    vtu_perm = torch.randperm(len(vtu_cell_position), generator=torch.Generator().manual_seed(run_id))
    vtu_perm = vtu_perm[: len(vtu_perm) // subsample_factor]

    # Compute distances from subsampled volume points to the surface mesh.
    subsampled_vtu_cell_position = vtu_cell_position[vtu_perm]
    if compute_surface_distance_via_trimesh:
        print(f"({run_folder}) Computing sdf via trimesh (this can take a while)")
        pq = trimesh.proximity.ProximityQuery(stl)
        # negative sign because outside by default is negative; clip to avoid negative distances (close to zero)
        position_dist_to_surface = (-pq.signed_distance(subsampled_vtu_cell_position)).clip(min=0)
    else:
        print(f"({run_folder}) Approximating surface distance via cKDTree")
        surface_tree = cKDTree(stl_resampled_pos)
        position_dist_to_surface, _ = surface_tree.query(subsampled_vtu_cell_position)
    assert len(position_dist_to_surface) == len(vtu_perm), (
        "Mismatch between number of volume cells and computed distances "
        f"({len(position_dist_to_surface)} != {len(vtu_perm)})"
    )

    print(f"({run_folder}) Storing volume data")
    torch.save(torch.Tensor(vtu_cell_position)[vtu_perm], run_dst_path / "volume_cell_position.pt")
    torch.save(torch.Tensor(vtu_cell_pressure)[vtu_perm], run_dst_path / "volume_cell_pressure.pt")
    torch.save(torch.Tensor(vtu_cell_velocity)[vtu_perm], run_dst_path / "volume_cell_velocity.pt")
    torch.save(torch.Tensor(vtu_cell_vorticity)[vtu_perm], run_dst_path / "volume_cell_vorticity.pt")
    torch.save(torch.Tensor(vtu_cell_pressure_total)[vtu_perm], run_dst_path / "volume_cell_totalpcoeff.pt")
    filename = f"volume_cell_surface_distance_{'trimesh' if compute_surface_distance_via_trimesh else 'cKDtree'}.pt"
    torch.save(torch.Tensor(position_dist_to_surface), run_dst_path / filename)  # is already subsampled

    print(f"({run_folder}) Finished processing run {run_id}.")


def main(
    dataset: str,
    root: str,
    output_dir: str,
    subsample_factor: int,
    compute_surface_distance_via_trimesh: bool,
    num_workers: int,
) -> None:
    assert dataset in dataset_keymaps, f"Unknown dataset {dataset}. Supported datasets: {list(dataset_keymaps.keys())}"
    data_field_keymap = dataset_keymaps.get(dataset)

    if dataset not in root:
        print(f"Warning: root={root} does not contain dataset={dataset}. Did you forget to adjust the source path?")
    if dataset not in output_dir:
        print(
            f"Warning: output_dir={output_dir} does not contain dataset={dataset}. Did you forget to adjust the destination path?"
        )

    # Check source/destination paths
    root_path = Path(root).expanduser()
    output_dir_path = Path(output_dir).expanduser()
    assert not output_dir_path.exists(), f"Destination folder already exists ('{output_dir_path.as_posix()}')"
    output_dir_path.mkdir()

    # Print configuration
    print(f"Dataset: {dataset}")
    print(f"root: {root_path.as_posix()}")
    print(f"output_dir: {output_dir_path.as_posix()}")
    print(f"subsample_factor: {subsample_factor}")
    print(f"compute_surface_distance_via_trimesh: {compute_surface_distance_via_trimesh}")

    # Find all run folders (assumed to be named "run_*")
    run_folders = [f for f in os.listdir(root_path) if f.startswith("run_")]
    print(f"Found {len(run_folders)} simulation(s)")

    # Sort by run_id (single digit, as its not necessarily sorted by default)
    run_folders = sorted(run_folders, key=lambda k: int(k.split("_")[1]))

    # Process all simulation files
    if num_workers > 1:
        print(f"Processing runs in parallel with {num_workers} workers...")
        tasks = [
            (
                run_folder,
                root_path,
                output_dir_path,
                data_field_keymap,
                subsample_factor,
                compute_surface_distance_via_trimesh,
            )
            for run_folder in run_folders
        ]
        with Pool(processes=num_workers) as pool:
            list(tqdm(pool.starmap(process_run, tasks, chunksize=1), total=len(tasks)))
    else:
        print("Processing runs sequentially in main process...")
        for run_folder in tqdm(run_folders):
            process_run(
                run_folder=run_folder,
                root=root_path,
                output_dir=output_dir_path,
                data_field_keymap=data_field_keymap,  # type: ignore[arg-type]
                subsample_factor=subsample_factor,
                compute_surface_distance_via_trimesh=compute_surface_distance_via_trimesh,
            )
    print("Finished processing all runs.")


if __name__ == "__main__":
    main(**parse_args())
