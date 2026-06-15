#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Writer that converts per-sample CFD tensors into a sharded, pre-shuffled Zarr store.

Each sample becomes an independent Zarr *group* (``<store_root>/<sample_id>.zarr``)
holding **one array per field** (``surface/position``, ``volume/velocity``, …), so
fields can be read independently. Points are shuffled once at write time with a
deterministic, per-sample seed so that any contiguous chunk is already a uniform-random
subset of the sample — this lets the dataloader turn "sample N random points" into
"read a random chunk". All arrays of a domain share the permutation and chunk grid, so
chunk ``c`` is point-aligned across fields.

Arrays are chunked along the point axis (``chunk_points``) with the channel axis left
unchunked, and packed into a single whole-array shard compressed per-chunk with
blosc+zstd — the per-sample object count therefore stays at one object per field.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from pathlib import Path

import numpy as np
import torch
import zarr
from zarr.codecs import BloscCodec

from noether.data.schemas import FileMap
from noether.data.zarr_store import stores
from noether.data.zarr_store.layout import build_domain_layouts
from noether.data.zarr_store.manifest import DomainSample, SampleEntry, StoreManifest

_INT64_MAX = 2**63 - 1


def _derive_seed(base_seed: int, sample_id: str, domain: str) -> int:
    """Deterministically derive a per-sample, per-domain shuffle seed."""
    digest = hashlib.sha256(f"{base_seed}:{sample_id}:{domain}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % _INT64_MAX


def _shuffle_perm(n: int, seed: int) -> np.ndarray:
    """Return a deterministic random permutation of ``range(n)``."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=generator).numpy()


class ZarrStoreWriter:
    """Convert CFD samples into the chunked/sharded Zarr format and track the manifest."""

    def __init__(
        self,
        store_root: str | Path,
        filemap: FileMap,
        dataset_name: str,
        shuffle_seed: int = 0,
        chunk_points: int = 4096,
        shard_points: int | None = None,
        coords_dtype: str = "float32",
        values_dtype: str = "float16",
        field_dtypes: dict[str, str] | None = None,
        compression_level: int = 5,
    ) -> None:
        """

        Args:
            store_root: Output location for the Zarr store. A local path or an fsspec URL
                (``s3://``, ``gs://``, ``memory://``, …) for object storage.
            filemap: Field-to-filename mapping describing which fields exist.
            dataset_name: Human-readable dataset name recorded in the manifest.
            shuffle_seed: Base seed for the per-sample point shuffle.
            chunk_points: Chunk size along the point axis. Pick close to the training
                subsample size to minimise read amplification.
            shard_points: Cap on the shard size along the point axis (rounded down to a
                whole number of chunks, minimum one chunk). ``None`` (default) packs each
                array into a single whole-array shard. Set this when per-field arrays grow
                large: shard bytes ≈ ``shard_points × dim × dtype_size``, so e.g. a
                ~128 MB cap on a float32×3 position array is ``shard_points ≈ 11_000_000``.
                Smaller shards bound the writer's per-shard RAM and the blast radius of a
                corrupt object, at the cost of more objects per array.
            coords_dtype: Dtype for the positions array (keep float32).
            values_dtype: Dtype for the physical fields array (float16 halves bytes).
            field_dtypes: Per-field dtype overrides keyed by canonical name, e.g.
                ``{"volume_vorticity": "float32"}`` for fields whose values exceed the
                ``values_dtype`` range (float16 caps at ~6.6e4); overflowing casts are
                rejected at write time rather than silently stored as ``inf``.
            compression_level: blosc/zstd compression level.
        """
        self.store_root = str(store_root)
        if not stores.is_remote(self.store_root):
            Path(self.store_root.removeprefix("file://")).mkdir(parents=True, exist_ok=True)
        if shard_points is not None and shard_points < 1:
            raise ValueError(f"shard_points must be positive, got {shard_points}.")
        self.filemap = filemap
        self.chunk_points = chunk_points
        self.shard_points = shard_points
        self.coords_dtype = coords_dtype
        self.values_dtype = values_dtype
        self.field_dtypes = field_dtypes
        self.compression_level = compression_level
        self._compressor = BloscCodec(cname="zstd", clevel=compression_level)

        self.layouts = build_domain_layouts(filemap, coords_dtype, values_dtype, field_dtypes)
        self.manifest = StoreManifest(
            dataset_name=dataset_name,
            shuffle_seed=shuffle_seed,
            coords_dtype=coords_dtype,
            values_dtype=values_dtype,
            compressor=f"blosc-zstd-{compression_level}",
            domains=self.layouts,
        )

    @staticmethod
    def _check_cast_range(field: str, data: np.ndarray, target_dtype: str) -> None:
        """Reject casts that would overflow *target_dtype* (silently storing ``inf``).

        Raises:
            ValueError: If finite values of *data* exceed the target float range.
        """
        target = np.dtype(target_dtype)
        if not (np.issubdtype(target, np.floating) and np.issubdtype(data.dtype, np.floating)):
            return
        limit = float(np.finfo(target).max)
        if float(np.finfo(data.dtype).max) <= limit:
            return  # widening or same-width cast cannot overflow
        peak = float(np.abs(data).max()) if data.size else 0.0
        if peak > limit:
            raise ValueError(
                f"Field '{field}' exceeds the {target_dtype} range (max |value| {peak:.4g} > {limit:.4g}); "
                f"store it at higher precision, e.g. field_dtypes={{'{field}': 'float32'}} "
                f"(CLI: --field-dtype {field}=float32)."
            )

    def _domain_grid(self, n_points: int) -> tuple[int, int, int]:
        """Compute a domain's ``(chunk, n_chunks, shard)`` along the point axis.

        The shard is the whole array (``n_chunks * chunk``) unless ``shard_points`` caps
        it, in which case it is the largest whole number of chunks that fits the cap
        (minimum one chunk) — Zarr requires shard shapes to be chunk multiples.
        """
        chunk = min(self.chunk_points, n_points)
        n_chunks = math.ceil(n_points / chunk)
        shard = n_chunks * chunk
        if self.shard_points is not None:
            shard = min(shard, max(1, self.shard_points // chunk) * chunk)
        return chunk, n_chunks, shard

    def _create_array(
        self, group: zarr.Group, name: str, n_points: int, width: int, dtype: str, chunk: int, shard: int
    ):
        """Create a sharded per-field array chunked along the point axis only."""
        return group.create_array(
            name=name,
            shape=(n_points, width),
            chunks=(chunk, width),
            shards=(shard, width),
            dtype=dtype,
            compressors=[self._compressor],
        )

    def write_group(self, sample_id: str, field_arrays: dict[str, np.ndarray]) -> SampleEntry:
        """Write one sample's Zarr group and return its manifest entry (no manifest mutation).

        Independent per sample (its own store), so this is safe to call concurrently from
        multiple threads; the caller records the returned entry in the manifest.

        Args:
            sample_id: Stable id used for the relative path and shuffle seed
                (e.g. ``"param1/<hash>"``).
            field_arrays: Mapping ``canonical_field -> numpy array``. Positions must be
                ``(N, 3)``; scalar fields may be ``(N,)`` or ``(N, 1)``.

        Returns:
            The :class:`SampleEntry` describing the written group.

        Raises:
            ValueError: If a domain's fields disagree on point count.
        """
        relpath = f"{sample_id}.zarr"
        store = stores.make_store(stores.join(self.store_root, relpath))
        group = zarr.create_group(store=store, overwrite=True)

        domain_samples: dict[str, DomainSample] = {}
        for domain, layout in self.layouts.items():
            if layout.position not in field_arrays:
                continue
            n_points = np.asarray(field_arrays[layout.position]).shape[0]
            perm = _shuffle_perm(n_points, _derive_seed(self.manifest.shuffle_seed, sample_id, domain))
            # One chunk/shard grid per domain so chunk c is point-aligned across all field arrays.
            chunk, n_chunks, shard = self._domain_grid(n_points)

            for field, array_layout in layout.arrays.items():
                data = np.asarray(field_arrays[field])
                if data.ndim == 1:
                    data = data[:, None]
                if data.shape[0] != n_points:
                    raise ValueError(f"Field '{field}' has {data.shape[0]} points, expected {n_points} for '{domain}'.")
                self._check_cast_range(field, data, array_layout.dtype)
                shuffled = np.ascontiguousarray(data[:, : array_layout.dim][perm], dtype=array_layout.dtype)
                array = self._create_array(
                    group, array_layout.array_name, n_points, array_layout.dim, array_layout.dtype, chunk, shard
                )
                array[:] = shuffled

            domain_samples[domain] = DomainSample(
                n_points=n_points, chunk_points=chunk, shard_points=shard, n_chunks=n_chunks
            )

        # Consolidate the group's metadata into its root zarr.json: opening a sample then
        # costs one metadata GET instead of one per group/array — a per-sample latency win
        # on object storage. Safe here because the store is write-once (no staleness), and
        # readers without consolidated-metadata support just fall back to per-node reads.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Consolidated metadata is currently not part")
            zarr.consolidate_metadata(store)

        return SampleEntry(relpath=relpath, domains=domain_samples)

    def write_sample(self, sample_id: str, field_arrays: dict[str, np.ndarray]) -> None:
        """Write one sample and record it in the manifest (sequential convenience)."""
        self.manifest.samples[sample_id] = self.write_group(sample_id, field_arrays)

    def to_init_kwargs(self) -> dict[str, object]:
        """Constructor kwargs to rebuild an identical writer (e.g. in a worker process)."""
        return {
            "store_root": self.store_root,
            "filemap": self.filemap,
            "dataset_name": self.manifest.dataset_name,
            "shuffle_seed": self.manifest.shuffle_seed,
            "chunk_points": self.chunk_points,
            "shard_points": self.shard_points,
            "coords_dtype": self.coords_dtype,
            "values_dtype": self.values_dtype,
            "field_dtypes": self.field_dtypes,
            "compression_level": self.compression_level,
        }

    def save_manifest(self) -> str:
        """Persist the manifest to the store root (local path or fsspec URL)."""
        return self.manifest.save(self.store_root)
