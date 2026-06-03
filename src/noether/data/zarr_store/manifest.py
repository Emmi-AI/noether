#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Manifest schema for the chunked/sharded Zarr CFD store.

The manifest is a small JSON sidecar written next to the per-sample Zarr groups.
It records the *global* column layout (which field lives in which array and at
which column offset) once, plus the *per-sample* chunk grid (point count, chunk
size, shard size, number of chunks) for every domain.

The dataloader uses the manifest to:

* pick random chunk indices per (sample, domain) without opening every array's
  metadata first, and
* map the fused value array columns back onto individual named fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import fsspec
from pydantic import BaseModel, Field


class ArrayLayout(BaseModel):
    """Layout of a single per-field Zarr array.

    Every field is its own array (``<domain>/<name>``) so fields can be read
    independently. The channel axis is never chunked, and each array is packed into a
    single whole-array shard, so the per-sample object count stays at one object per
    field while chunks remain individually range-readable.
    """

    array_name: str
    """Zarr path of the array within the per-sample group, e.g. ``"volume/velocity"``."""
    field: str
    """Canonical field name served by this array, e.g. ``"volume_velocity"``."""
    dtype: str
    """On-disk dtype of the array, e.g. ``"float32"`` or ``"float16"``."""
    dim: int
    """Channel width (1 for scalars, 3 for vectors)."""


class DomainLayout(BaseModel):
    """Per-field arrays of one domain.

    All arrays of a domain share the point axis, the shuffle permutation and the chunk
    grid, so chunk ``c`` addresses the same physical points in every field.
    """

    position: str
    """Canonical name of the domain's coordinate field (e.g. ``"volume_position"``)."""
    arrays: dict[str, ArrayLayout]
    """Mapping ``canonical_field -> array layout`` (includes the position array)."""


class DomainSample(BaseModel):
    """Per-sample chunk grid for one domain."""

    n_points: int
    """Number of points (rows) for this sample/domain."""
    chunk_points: int
    """Chunk size along the point axis."""
    shard_points: int
    """Shard size along the point axis — a whole number of chunks; the full array
    (``n_chunks * chunk_points``) unless the writer's ``shard_points`` cap split the
    arrays into multiple shards."""
    n_chunks: int
    """Number of chunks along the point axis (``ceil(n_points / chunk_points)``)."""


class SampleEntry(BaseModel):
    """Manifest entry for a single sample."""

    relpath: str
    """Path of the sample's Zarr group relative to the store root."""
    domains: dict[str, DomainSample]
    """Per-domain chunk grids."""


class StoreManifest(BaseModel):
    """Top-level manifest for a converted Zarr store."""

    dataset_name: str
    format_version: int = 1
    shuffle_seed: int
    """Base seed used to derive the per-sample point shuffle permutations."""
    coords_dtype: str = "float32"
    values_dtype: str = "float16"
    compressor: str = "blosc-zstd"
    domains: dict[str, DomainLayout]
    """Global column layout, shared by every sample."""
    samples: dict[str, SampleEntry] = Field(default_factory=dict)
    """Per-sample chunk grids keyed by sample id."""

    MANIFEST_NAME: ClassVar[str] = "manifest.json"

    def save(self, store_root: str | Path) -> str:
        """Write the manifest to ``<store_root>/manifest.json`` (local path or fsspec URL)."""
        path = f"{str(store_root).rstrip('/')}/{self.MANIFEST_NAME}"
        with fsspec.open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, store_root: str | Path) -> StoreManifest:
        """Load the manifest from ``<store_root>/manifest.json`` (local path or fsspec URL)."""
        path = f"{str(store_root).rstrip('/')}/{cls.MANIFEST_NAME}"
        with fsspec.open(path, "r") as f:
            return cls.model_validate_json(f.read())
