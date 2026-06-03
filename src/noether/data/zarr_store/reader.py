#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Chunk-sampling reader for the Zarr CFD store.

The reader turns "draw ``T`` random points from a sample" into "read a few random
chunks". Because points were shuffled once at write time, any chunk is already a
uniform-random subset, so the reader only fetches ``ceil(T / chunk_points)`` chunks
per array instead of the whole sample. Reading a chunk from a sharded array is a
byte-range request, so the bytes transferred scale with ``T`` and not with the
sample size.

The per-sample chunk reads are independent, so they can be issued concurrently. Set
``read_concurrency > 1`` to fetch a sample's chunks (across both arrays and both
domains) in a thread pool — this hides per-request latency on high-latency stores
(e.g. S3) and is a no-op cost on fast local stores, where the default of ``1`` keeps
reads serial.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import numpy as np
import torch
import zarr
from zarr.abc.store import Store

from noether.data.zarr_store import stores
from noether.data.zarr_store.manifest import ArrayLayout, StoreManifest

# One per-array read job: (array, chunk-aligned row slices or None for full, target points, layout).
_ReadJob = tuple["zarr.Array", "list[tuple[int, int]] | None", "int | None", ArrayLayout]


def _default_store_factory(path: str) -> Store:
    """Open a read-only store for a sample group (LocalStore or FsspecStore by path)."""
    return stores.make_store(path, read_only=True)


class ZarrChunkReader:
    """Reads chunk-subsampled (or full) per-field tensors from a converted store."""

    def __init__(
        self,
        store_root: str | Path,
        manifest: StoreManifest | None = None,
        store_factory: Callable[[str], Store] = _default_store_factory,
        read_concurrency: int = 1,
    ) -> None:
        """

        Args:
            store_root: Root of the converted Zarr store — a local path or an fsspec URL
                (``s3://``, ``gs://``, ``memory://``, …) for object storage.
            manifest: Pre-loaded manifest; loaded from ``store_root`` if omitted.
            store_factory: Builds a Zarr store from a sample-group path/URL. Defaults to a
                read-only LocalStore/FsspecStore by path; override to wrap the store (e.g.
                for byte-accounting in benchmarks).
            read_concurrency: Max threads used to fetch a sample's chunks in parallel.
                ``1`` (default) reads serially — best for fast local stores. Use a value
                around the number of chunks per sample to hide latency on S3.
        """
        self.store_root = str(store_root)
        self.manifest = manifest if manifest is not None else StoreManifest.load(self.store_root)
        self._store_factory = store_factory
        self.read_concurrency = max(1, read_concurrency)
        self._groups: dict[str, zarr.Group] = {}

    def _group(self, relpath: str) -> zarr.Group:
        """Return a cached read-only handle to a sample's Zarr group."""
        group = self._groups.get(relpath)
        if group is None:
            group = zarr.open_group(store=self._store_factory(stores.join(self.store_root, relpath)), mode="r")
            self._groups[relpath] = group
        return group

    @staticmethod
    def _select_row_slices(
        n_points: int,
        chunk_points: int,
        n_chunks: int,
        num_points: int | None,
        generator: torch.Generator | None,
    ) -> list[tuple[int, int]] | None:
        """Pick chunk-aligned ``(start, stop)`` row slices covering ``num_points`` rows.

        Returns ``None`` to signal "read everything" (when ``num_points`` is ``None`` or
        at least the sample size). Otherwise random chunk indices are accumulated until
        their combined length reaches ``num_points``; the caller trims the surplus.
        """
        if num_points is None or num_points >= n_points:
            return None
        order = torch.randperm(n_chunks, generator=generator).tolist()
        slices: list[tuple[int, int]] = []
        gathered = 0
        for chunk_idx in order:
            start = chunk_idx * chunk_points
            stop = min(start + chunk_points, n_points)
            slices.append((start, stop))
            gathered += stop - start
            if gathered >= num_points:
                break
        return slices

    @staticmethod
    def _assemble(parts: list[np.ndarray], slices: list[tuple[int, int]] | None, num_points: int | None) -> np.ndarray:
        """Concatenate a single array's chunk blocks (in order) and trim to ``num_points``."""
        if slices is None:
            return parts[0]
        data = np.concatenate(parts, axis=0)
        if num_points is not None:
            data = data[:num_points]
        return cast("np.ndarray", data)

    @staticmethod
    def _row_indices(slices: list[tuple[int, int]], num_points: int | None) -> np.ndarray:
        """Flatten chunk-aligned slices into an ordered row-index array, trimmed to ``num_points``."""
        rows = np.concatenate([np.arange(start, stop) for start, stop in slices])
        return cast("np.ndarray", rows[:num_points] if num_points is not None else rows)

    @staticmethod
    def _to_field_tensor(layout: ArrayLayout, data: np.ndarray) -> torch.Tensor:
        """Convert one per-field array block to its float32 tensor (scalars squeezed to 1-D)."""
        tensor = torch.from_numpy(np.ascontiguousarray(data)).float()
        return tensor.squeeze(1) if layout.dim == 1 else tensor

    def _read_serial(self, jobs: list[_ReadJob]) -> dict[int, np.ndarray]:
        """One read per array — a single orthogonal selection (or full read).

        Lowest call overhead, so this is the fastest path on low-latency stores (local
        disk / NFS). zarr fetches the chunks the selection touches.
        """
        blocks: dict[int, np.ndarray] = {}
        for job_idx, (array, slices, target, _) in enumerate(jobs):
            if slices is None:
                blocks[job_idx] = np.asarray(array[:])
            else:
                blocks[job_idx] = np.asarray(array.oindex[self._row_indices(slices, target)])
        return blocks

    def _read_parallel(self, jobs: list[_ReadJob]) -> dict[int, np.ndarray]:
        """Flatten every ``(array, chunk)`` read into one concurrent batch.

        Maximises overlap across both arrays and chunks, which hides per-request latency
        on high-latency stores (S3). Each chunk is a contiguous single-chunk slice read.
        """
        tasks: list[tuple[int, int, zarr.Array, int | None, int | None]] = []
        for job_idx, (array, slices, _, _) in enumerate(jobs):
            if slices is None:
                tasks.append((job_idx, 0, array, None, None))
            else:
                for part_idx, (start, stop) in enumerate(slices):
                    tasks.append((job_idx, part_idx, array, start, stop))

        def _run(task: tuple[int, int, zarr.Array, int | None, int | None]) -> tuple[int, int, np.ndarray]:
            job_idx, part_idx, array, start, stop = task
            block = np.asarray(array[:] if start is None else array[start:stop])
            return job_idx, part_idx, block

        with ThreadPoolExecutor(max_workers=min(self.read_concurrency, len(tasks))) as executor:
            results = list(executor.map(_run, tasks))

        parts_by_job: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
        for job_idx, part_idx, block in results:
            parts_by_job[job_idx].append((part_idx, block))

        blocks: dict[int, np.ndarray] = {}
        for job_idx, (_, slices, target, _) in enumerate(jobs):
            ordered = [block for _, block in sorted(parts_by_job[job_idx])]
            blocks[job_idx] = self._assemble(ordered, slices, target)
        return blocks

    def read_sample(
        self,
        sample_id: str,
        num_points: dict[str, int | None] | None = None,
        generator: torch.Generator | None = None,
        fields: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Read (optionally chunk-subsampled) fields for one sample.

        With ``read_concurrency == 1`` each array is read with a single orthogonal
        selection (lowest overhead, best on local stores). With ``read_concurrency > 1``
        all chunk reads of the sample are issued concurrently in a thread pool (best
        latency hiding on S3). Both paths return identical results.

        Args:
            sample_id: Sample id present in the manifest.
            num_points: Per-domain target counts, e.g. ``{"surface": 3586, "volume": 4096}``.
                A ``None`` value (or missing domain) reads the full domain.
            generator: Torch RNG for chunk selection. Pass a seeded generator for
                deterministic evaluation; ``None`` draws a fresh subset each call.
            fields: Optional subset of canonical fields to read. Because every field is
                its own array, unrequested fields cost no I/O. ``None`` reads all fields.

        Returns:
            Mapping ``canonical_field -> tensor`` with scalar fields shaped ``(T,)`` and
            vector fields shaped ``(T, dim)``, all float32 and point-aligned per domain.
        """
        num_points = num_points or {}
        entry = self.manifest.samples[sample_id]
        group = self._group(entry.relpath)

        # Plan one read job per (requested) field array; all arrays of a domain share the
        # same chunk selection so their rows stay point-aligned.
        jobs: list[_ReadJob] = []
        for domain, layout in self.manifest.domains.items():
            if domain not in entry.domains:
                continue
            wanted = [al for field, al in layout.arrays.items() if fields is None or field in fields]
            if not wanted:
                continue
            grid = entry.domains[domain]
            target = num_points.get(domain)
            slices = self._select_row_slices(grid.n_points, grid.chunk_points, grid.n_chunks, target, generator)
            jobs.extend((cast("zarr.Array", group[al.array_name]), slices, target, al) for al in wanted)

        blocks = self._read_parallel(jobs) if self.read_concurrency > 1 else self._read_serial(jobs)

        out: dict[str, torch.Tensor] = {}
        for job_idx, (_, _, _, array_layout) in enumerate(jobs):
            out[array_layout.field] = self._to_field_tensor(array_layout, blocks[job_idx])
        return out

    def read_coords(
        self,
        sample_id: str,
        domain: str,
        num_points: int | None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Read an independent chunk-subsample of a domain's point positions only.

        This is a separate random draw from :meth:`read_sample` (it consumes the same
        generator, so it is uncorrelated with the field draw), reading just the domain's
        position array. It backs the AB-UPT ``geometry_position`` input — a random draw
        of the surface points distinct from the surface anchor points.

        Args:
            sample_id: Sample id present in the manifest.
            domain: Domain whose positions to read (e.g. ``"surface"``).
            num_points: Number of points to sample (``None`` reads all positions).
            generator: Torch RNG for chunk selection.

        Returns:
            ``(num_points, position_dim)`` float32 positions tensor.
        """
        entry = self.manifest.samples[sample_id]
        layout = self.manifest.domains[domain]
        position_layout = layout.arrays[layout.position]
        grid = entry.domains[domain]
        group = self._group(entry.relpath)
        slices = self._select_row_slices(grid.n_points, grid.chunk_points, grid.n_chunks, num_points, generator)
        jobs: list[_ReadJob] = [
            (cast("zarr.Array", group[position_layout.array_name]), slices, num_points, position_layout)
        ]
        blocks = self._read_parallel(jobs) if self.read_concurrency > 1 else self._read_serial(jobs)
        return self._to_field_tensor(position_layout, blocks[0])
