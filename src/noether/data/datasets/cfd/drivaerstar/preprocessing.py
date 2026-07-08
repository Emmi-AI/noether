#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Process raw DrivAerStar EnSight cases and convert them into a chunked Zarr store.

DrivAerStar ships each simulation as an EnSight Gold case (``<T>_<id>.case`` + ``.geo`` +
per-cell ``Pressure`` / ``Velocity`` / ``WallShearStress`` files) under
``case_<T>/<id>/case/``. This is the DrivAerStar analogue of
:mod:`noether.data.datasets.cfd.caeml.preprocessing`, but instead of writing per-field
``.pt`` files it writes directly into the blob-storage-friendly chunked/sharded Zarr store
(:class:`~noether.data.zarr_store.writer.ZarrStoreWriter`), so the result is consumable by
:class:`~noether.data.datasets.cfd.ZarrAeroDataset` and friends.

Surface / volume split follows the convention used in ``Emmi-AI/proprioceptive``'s
``match_experiment_probes.py``:

* **volume** — the ``domain`` block (Pressure + Velocity per cell);
* **surface** — every ``WallShearStress``-bearing block that is *not* the volume and *not*
  a wind-tunnel wall (``Block.*``, including the ground plane ``Block.bottom``); i.e. the
  car / wheel shell. Combined and surface-extracted, fields taken at cell centres.

For each case the script stores (canonical field → array):

* ``surface_position`` / ``surface_pressure`` / ``surface_friction`` (WallShearStress) /
  ``surface_normals``;
* ``volume_position`` / ``volume_pressure`` / ``volume_velocity``;
* ``volume_sdf`` (distance from each volume cell centre to the car surface) — **only when
  ``--sdf`` is passed**; off by default since it needs scipy and a KD-tree query over all
  volume points.

Robustness: some DrivAerStar binaries crash VTK's EnSight reader at the C level
(uncatchable from Python), so each case is read/extracted in a *child process*; a crashing
case yields a non-zero exit and is skipped with a warning rather than killing the run.

Source may be local or any fsspec URL (the bucket lives at
``oci://emmi-drivaer-star``; ``ocifs`` needs ``OCIFS_IAM_TYPE=api_key``)::

    OCIFS_IAM_TYPE=api_key uv run python -m noether.data.datasets.cfd.drivaerstar.preprocessing \\
        --source oci://emmi-drivaer-star@frwnorq7ern2 \\
        --output oci://emmi-drivaer-star@frwnorq7ern2/zarr_store \\
        --case-types E --workers 16
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

from noether.data.schemas import FileMap
from noether.data.zarr_store import stores
from noether.data.zarr_store.manifest import SampleEntry
from noether.data.zarr_store.writer import ZarrStoreWriter


# Canonical field -> placeholder filename. The Zarr layout only uses the FileMap to decide
# which fields exist and their canonical names (``volume_distance_to_surface`` -> ``volume_sdf``);
# the filenames are what a downstream ``AeroDataset`` would map back from, so keep them stable.
def build_filemap(compute_sdf: bool = False) -> FileMap:
    """Return the DrivAerStar FileMap, optionally including the volume SDF field.

    Args:
        compute_sdf: If ``True``, include ``volume_distance_to_surface`` (canonical
            ``volume_sdf``); when ``False`` it is omitted so the writer does not expect it.
    """
    return FileMap(
        surface_position="surface_position.pt",
        surface_pressure="surface_pressure.pt",
        surface_friction="surface_wallshearstress.pt",
        surface_normals="surface_normals.pt",
        volume_position="volume_position.pt",
        volume_pressure="volume_pressure.pt",
        volume_velocity="volume_velocity.pt",
        volume_distance_to_surface="volume_sdf.pt" if compute_sdf else None,
    )


# Full field set (SDF included); use :func:`build_filemap` to opt out of the SDF.
DRIVAERSTAR_FILEMAP = build_filemap(compute_sdf=True)

# Long-edge human label per case-type prefix (DrivAerStar Estate/Fastback/Notchback).
VARIANT_BY_TYPE = {"E": "ESTATEBACK", "F": "FASTBACK", "N": "NOTCHBACK"}

# EnSight sibling files that make up one case (relative to the ``.case`` file).
_CASE_SUFFIXES = (".case", ".geo", ".Pressure", ".Velocity", ".WallShearStress")


# --------------------------------------------------------------------------------------
# Field extraction (runs in a child process, see ``_extract_main``)
# --------------------------------------------------------------------------------------
def extract_case_fields(
    case_path: str | Path, *, subsample_factor: int = 1, compute_sdf: bool = False
) -> dict[str, np.ndarray]:
    """Read one EnSight case and return canonical field arrays for surface and volume.

    Args:
        case_path: Path to a local ``<T>_<id>.case`` file (siblings ``.geo`` / ``.Pressure`` /
            ``.Velocity`` / ``.WallShearStress`` must be next to it).
        subsample_factor: Keep ``1 / subsample_factor`` of the volume cells via a seeded
            permutation (``1`` keeps all). Surface cells are always kept in full.
        compute_sdf: If ``True``, also compute ``volume_sdf`` (distance from each volume cell
            centre to the nearest car-surface vertex). Off by default — it needs scipy and
            adds a KD-tree query over all volume points.

    Returns:
        Mapping ``canonical_field -> np.ndarray`` (positions ``float32 (N, 3)``, scalar
        fields ``(N,)``, vector fields ``(N, 3)``), ready for
        :meth:`~noether.data.zarr_store.writer.ZarrStoreWriter.write_group`.

    Raises:
        RuntimeError: If the case has no ``domain`` block or no WallShearStress surface.
    """
    import pyvista as pv

    mb = pv.read(str(case_path))

    volume = None
    wss_parts: list = []  # WallShearStress-bearing blocks = car / wheel surfaces
    for i in range(mb.n_blocks):
        block = mb.get_block(i)
        if block is None or block.n_cells == 0 or block.n_points == 0:
            continue
        name = (mb.get_block_name(i) or "").strip().lower()
        if name == "domain":
            volume = block
            continue
        if name.startswith("block."):
            continue  # wind-tunnel walls / ground plane
        if "WallShearStress" not in list(block.array_names):
            continue  # porosity helpers, interface patches, ...
        wss_parts.append(block)

    if volume is None:
        raise RuntimeError(f"no 'domain' (volume) block in {case_path}")
    if not wss_parts:
        raise RuntimeError(f"no WallShearStress surface blocks in {case_path}")

    combined = wss_parts[0] if len(wss_parts) == 1 else pv.MultiBlock(wss_parts).combine()
    surface = combined.extract_surface(algorithm="dataset_surface")
    surface_normals = surface.compute_normals(
        cell_normals=True, point_normals=False, auto_orient_normals=True, consistent_normals=True
    )

    fields: dict[str, np.ndarray] = {
        "surface_position": np.asarray(surface.cell_centers().points, dtype=np.float32),
        "surface_pressure": np.asarray(surface.cell_data["Pressure"], dtype=np.float32),
        "surface_friction": np.asarray(surface.cell_data["WallShearStress"], dtype=np.float32),
        "surface_normals": np.asarray(surface_normals.cell_data["Normals"], dtype=np.float32),
    }

    volume_position = np.asarray(volume.cell_centers().points, dtype=np.float32)
    volume_pressure = np.asarray(volume.cell_data["Pressure"], dtype=np.float32)
    volume_velocity = np.asarray(volume.cell_data["Velocity"], dtype=np.float32)
    if subsample_factor > 1:
        # Seeded so a re-run reproduces the same volume subset (matches caeml's convention).
        import torch

        seed = abs(hash(str(case_path))) % (2**31)
        perm = torch.randperm(len(volume_position), generator=torch.Generator().manual_seed(seed)).numpy()
        perm = perm[: len(perm) // subsample_factor]
        volume_position, volume_pressure, volume_velocity = (
            volume_position[perm],
            volume_pressure[perm],
            volume_velocity[perm],
        )

    fields.update(
        volume_position=volume_position,
        volume_pressure=volume_pressure,
        volume_velocity=volume_velocity,
    )

    if compute_sdf:
        # Distance from each volume cell centre to the nearest car-surface vertex, via a KD-tree
        # (the same approximation caeml's preprocessing uses — cheap and accurate to ~mm, vs.
        # VTK's exact point-to-mesh distance which is ~1000x slower on ~10M points).
        try:
            from scipy.spatial import cKDTree
        except ImportError as exc:  # pragma: no cover - optional preprocessing dependency
            raise RuntimeError(
                "DrivAerStar SDF (--sdf) needs scipy; install it (e.g. `uv add scipy` or the 'preprocessing' extra)."
            ) from exc

        tree = cKDTree(np.asarray(surface.points, dtype=np.float32))
        volume_sdf, _ = tree.query(volume_position, workers=1)
        fields["volume_sdf"] = volume_sdf.astype(np.float32)

    return fields


def _extract_main() -> int:
    """Child-process entry point: ``--extract <case_path> --out <npz> [--subsample-factor N] [--sdf]``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", required=True, help="Path to the local .case file.")
    parser.add_argument("--out", required=True, help="Where to write the .npz of field arrays.")
    parser.add_argument("--subsample-factor", type=int, default=1)
    parser.add_argument("--sdf", action="store_true", help="Also compute the volume SDF field.")
    args = parser.parse_args(sys.argv[1:])
    fields = extract_case_fields(args.extract, subsample_factor=args.subsample_factor, compute_sdf=args.sdf)
    # numpy's savez stub mistypes the first **kwarg as ``allow_pickle: bool``; our kwargs are arrays.
    np.savez(args.out, **fields)  # type: ignore[arg-type]
    return 0


# --------------------------------------------------------------------------------------
# Case discovery + per-case driver (parent process)
# --------------------------------------------------------------------------------------
def discover_cases(source: str, case_types: list[str], limit: int | None = None) -> list[tuple[str, str, str]]:
    """Find DrivAerStar cases under ``source`` for the requested case types.

    Args:
        source: Dataset root (local path or fsspec URL) holding ``case_<T>/<id>/case/...``.
        case_types: Case-type prefixes to include (subset of ``E`` / ``F`` / ``N``).
        limit: If set, keep at most this many cases (after sorting) across all types.

    Returns:
        Sorted list of ``(sample_id, case_type, case_url)`` where ``sample_id`` is
        ``"<T>_<id>"`` and ``case_url`` points at the ``.case`` file (local path or URL).
    """
    cases: list[tuple[str, str, str]] = []
    if stores.is_remote(source):
        import fsspec

        fs = fsspec.filesystem(source.split("://", 1)[0])
        root = source.split("://", 1)[1]
        for case_type in case_types:
            pattern = f"{root.rstrip('/')}/case_{case_type}/*/case/{case_type}_*.case"
            for hit in fs.glob(pattern):
                case_id = Path(hit).stem.split("_", 1)[1]
                scheme = source.split("://", 1)[0]
                cases.append((f"{case_type}_{case_id}", case_type, f"{scheme}://{hit}"))
    else:
        base = Path(source)
        for case_type in case_types:
            for hit in sorted((base / f"case_{case_type}").glob(f"*/case/{case_type}_*.case")):
                case_id = hit.stem.split("_", 1)[1]
                cases.append((f"{case_type}_{case_id}", case_type, str(hit)))

    cases.sort(key=lambda c: c[0])
    return cases[:limit] if limit else cases


def _materialize_case(case_url: str, tmp_dir: Path) -> Path:
    """Ensure a case's EnSight files are on local disk; return the local ``.case`` path.

    Local sources are used in place; remote sources have their sibling EnSight files
    downloaded into ``tmp_dir`` via fsspec.
    """
    if not stores.is_remote(case_url):
        return Path(case_url)

    import fsspec

    scheme, path = case_url.split("://", 1)
    fs = fsspec.filesystem(scheme)
    stem = Path(path).name.rsplit(".case", 1)[0]  # e.g. "E_03210"
    remote_dir = path.rsplit("/", 1)[0]
    for suffix in _CASE_SUFFIXES:
        remote = f"{remote_dir}/{stem}{suffix}"
        if fs.exists(remote):
            fs.get_file(remote, str(tmp_dir / f"{stem}{suffix}"))
    return tmp_dir / f"{stem}.case"


def convert_case(
    sample_id: str,
    case_url: str,
    writer: ZarrStoreWriter,
    *,
    subsample_factor: int = 1,
    compute_sdf: bool = False,
    timeout_s: int = 1800,
) -> SampleEntry | None:
    """Extract one case (in a child process) and write its Zarr group; ``None`` on failure.

    The EnSight read + field extraction run in a separate interpreter so a C-level crash in
    VTK's reader fails just this case (non-zero exit) instead of the whole run.

    Args:
        sample_id: Stable id (``"<T>_<id>"``) used for the store path and shuffle seed.
        case_url: Local path or fsspec URL of the ``.case`` file.
        writer: Target store writer (its FileMap defines which fields are stored).
        subsample_factor: Volume subsampling factor passed to the child (``1`` keeps all).
        compute_sdf: Whether the child should also compute the ``volume_sdf`` field.
        timeout_s: Per-case wall-clock budget for the extraction child.

    Returns:
        The :class:`SampleEntry` for the written group, or ``None`` if the case was skipped.
    """
    with tempfile.TemporaryDirectory(prefix="drivaerstar_") as tmp:
        tmp_dir = Path(tmp)
        try:
            local_case = _materialize_case(case_url, tmp_dir)
            out_npz = tmp_dir / f"{sample_id}.npz"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "noether.data.datasets.cfd.drivaerstar.preprocessing",
                    "--extract",
                    str(local_case),
                    "--out",
                    str(out_npz),
                    "--subsample-factor",
                    str(subsample_factor),
                    *(["--sdf"] if compute_sdf else []),
                ],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            if proc.returncode != 0 or not out_npz.exists():
                warnings.warn(
                    f"Skipping '{sample_id}': extraction failed (exit {proc.returncode}). {proc.stderr.strip()[-500:]}",
                    stacklevel=2,
                )
                return None
            with np.load(out_npz) as data:
                fields = {key: data[key] for key in data.files}
            return writer.write_group(sample_id, fields)
        except subprocess.TimeoutExpired:
            warnings.warn(f"Skipping '{sample_id}': extraction timed out after {timeout_s}s.", stacklevel=2)
            return None
        except Exception as exc:  # noqa: BLE001 - one bad case must not abort the run
            warnings.warn(f"Skipping '{sample_id}': {type(exc).__name__}: {exc}", stacklevel=2)
            return None


def convert(
    source: str,
    output: str,
    *,
    case_types: list[str],
    workers: int = 1,
    subsample_factor: int = 1,
    compute_sdf: bool = False,
    chunk_points: int = 4096,
    shard_points: int | None = None,
    shuffle_seed: int = 0,
    values_dtype: str = "float16",
    field_dtypes: dict[str, str] | None = None,
    limit: int | None = None,
    timeout_s: int = 1800,
) -> ZarrStoreWriter:
    """Convert raw DrivAerStar EnSight cases into a chunked Zarr store.

    Cases are extracted concurrently in up to ``workers`` child processes (crash-isolated);
    completed cases are written into the store and recorded in the manifest as they finish.
    Failed cases are skipped with a warning. The manifest is saved once at the end.

    Args:
        source: Dataset root (local path or fsspec URL) with ``case_<T>/<id>/case/...``.
        output: Zarr store root (local path or fsspec URL).
        case_types: Case-type prefixes to convert (subset of ``E`` / ``F`` / ``N``).
        workers: Maximum number of concurrent extraction child processes.
        subsample_factor: Volume subsampling factor (``1`` keeps all ~10M cells).
        compute_sdf: Whether to also compute and store the ``volume_sdf`` field (off by default).
        chunk_points: Points per chunk along the point axis.
        shard_points: Optional shard-size cap (point-multiple of ``chunk_points``).
        shuffle_seed: Base seed for the per-sample point shuffle.
        values_dtype: Default dtype for value fields (positions are always float32).
        field_dtypes: Per-canonical-field dtype overrides, e.g. ``{"volume_pressure": "float32"}``.
        limit: Convert at most this many cases.
        timeout_s: Per-case extraction timeout.

    Returns:
        The :class:`ZarrStoreWriter` after all cases are written and the manifest saved.

    Raises:
        RuntimeError: If no cases are found, or every discovered case failed to convert.
    """
    cases = discover_cases(source, case_types, limit=limit)
    if not cases:
        raise RuntimeError(f"No DrivAerStar cases found under '{source}' for types {case_types}.")
    print(f"Found {len(cases)} case(s) for types {case_types}; converting with {workers} worker(s).")

    writer = ZarrStoreWriter(
        store_root=output,
        filemap=build_filemap(compute_sdf=compute_sdf),
        dataset_name="drivaerstar",
        shuffle_seed=shuffle_seed,
        chunk_points=chunk_points,
        shard_points=shard_points,
        values_dtype=values_dtype,
        field_dtypes=field_dtypes,
    )

    # Bounded scheduler: each case is extracted by its own interpreter (crash isolation),
    # at most ``workers`` at a time, and written to the store from the parent as it finishes.
    converted = 0
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Threads only orchestrate subprocesses + the (GIL-light) zarr writes, so a thread pool
    # is enough to overlap extraction and I/O without the BrokenProcessPool crash hazard.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                convert_case,
                sample_id,
                case_url,
                writer,
                subsample_factor=subsample_factor,
                compute_sdf=compute_sdf,
                timeout_s=timeout_s,
            ): sample_id
            for sample_id, _case_type, case_url in cases
        }
        for done in as_completed(futures):
            sample_id = futures[done]
            entry = done.result()
            if entry is not None:
                writer.manifest.samples[sample_id] = entry
                converted += 1
                print(f"[{converted}/{len(cases)}] wrote {sample_id}")

    if converted == 0:
        raise RuntimeError("All discovered cases failed to convert.")
    path = writer.save_manifest()
    print(f"Converted {converted}/{len(cases)} case(s). Manifest: {path}")
    return writer


def _parse_field_dtypes(values: list[str] | None) -> dict[str, str] | None:
    if not values:
        return None
    out: dict[str, str] = {}
    for item in values:
        field, _, dtype = item.partition("=")
        if not dtype:
            raise ValueError(f"--field-dtype expects FIELD=DTYPE, got '{item}'")
        out[field] = dtype
    return out


def main() -> None:
    """CLI entry point for DrivAerStar EnSight -> Zarr conversion."""
    # Child-process fast path: ``--extract`` is handled before the full CLI is built.
    if "--extract" in sys.argv:
        raise SystemExit(_extract_main())

    parser = argparse.ArgumentParser("Convert raw DrivAerStar EnSight cases into a chunked Zarr store.")
    parser.add_argument("--source", required=True, help="Dataset root (local path or fsspec URL).")
    parser.add_argument("--output", required=True, help="Zarr store root (local path or fsspec URL).")
    parser.add_argument(
        "--case-types",
        nargs="+",
        default=["E", "F", "N"],
        choices=["E", "F", "N"],
        help="Case-type prefixes to convert (Estate / Fastback / Notchback).",
    )
    parser.add_argument("--workers", type=int, default=1, help="Concurrent extraction child processes.")
    parser.add_argument("--subsample-factor", type=int, default=1, help="Keep 1/factor of volume cells (1 = keep all).")
    parser.add_argument(
        "--sdf",
        action="store_true",
        help="Also compute the volume SDF (distance to surface); off by default. Needs scipy.",
    )
    parser.add_argument("--chunk-points", type=int, default=4096, help="Points per chunk.")
    parser.add_argument("--shard-points", type=int, default=None, help="Optional shard-size cap (point count).")
    parser.add_argument("--shuffle-seed", type=int, default=0, help="Base seed for the per-sample point shuffle.")
    parser.add_argument("--values-dtype", default="float16", help="Default dtype for value fields.")
    parser.add_argument(
        "--field-dtype",
        action="append",
        default=None,
        metavar="FIELD=DTYPE",
        help="Per-field dtype override (repeatable), e.g. --field-dtype volume_pressure=float32.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Convert at most this many cases.")
    parser.add_argument("--timeout-s", type=int, default=1800, help="Per-case extraction timeout in seconds.")
    args = parser.parse_args()

    convert(
        source=args.source,
        output=args.output,
        case_types=args.case_types,
        workers=args.workers,
        subsample_factor=args.subsample_factor,
        compute_sdf=args.sdf,
        chunk_points=args.chunk_points,
        shard_points=args.shard_points,
        shuffle_seed=args.shuffle_seed,
        values_dtype=args.values_dtype,
        field_dtypes=_parse_field_dtypes(args.field_dtype),
        limit=args.limit,
        timeout_s=args.timeout_s,
    )


if __name__ == "__main__":
    # Avoid oversubscription when many workers each spawn a VTK-heavy child.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
