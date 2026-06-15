#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Zarr-backed :class:`AeroDataset` that subsamples by reading chunks.

This is the read side of the chunked/sharded Zarr format. Instead of loading every
field of a sample and discarding most points (the ``.pt`` + ``PointSamplingSampleProcessor``
path), the dataset reads only the random chunks it needs:

* :meth:`pre_getitem` selects random chunks per domain and fetches just those rows
  (byte-range reads against the sharded arrays), splitting the fused arrays back into
  per-field tensors.
* the inherited ``getitem_*`` / ``with_normalizers`` machinery then serves those
  pre-read tensors, so normalization, key names and downstream collation are unchanged.

Set ``num_points`` to ``None`` per domain (the default) to read full samples — e.g. for
evaluation — or to an integer to chunk-subsample at read time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from noether.core.utils.common import validate_path
from noether.data.base.dataset import StandardDatasetConfig, with_normalizers
from noether.data.datasets.cfd.dataset import AeroDataset
from noether.data.schemas import FileMap
from noether.data.zarr_store import stores
from noether.data.zarr_store.layout import filename_to_canonical
from noether.data.zarr_store.manifest import StoreManifest
from noether.data.zarr_store.reader import ZarrChunkReader

# getitem_* methods that do not correspond to a stored field (computed/derived).
_COMPUTED_GETITEMS = {"getitem_surface_sdf"}

# Domain the AB-UPT geometry input is drawn from (a random draw of the car surface).
_GEOMETRY_SOURCE_DOMAIN = "surface"


class ZarrAeroDatasetConfig(StandardDatasetConfig):
    """Config for Zarr-backed aerodynamic datasets with chunk-based subsampling.

    ``root`` points at the converted Zarr store. Leave the ``num_*`` fields ``None`` to
    read full samples (e.g. evaluation); set them to chunk-subsample at read time, in
    which case the pipeline's ``PointSamplingSampleProcessor`` becomes a no-op.
    """

    num_surface_points: int | None = None
    """Surface points to chunk-sample per item (``None`` = full surface)."""
    num_volume_points: int | None = None
    """Volume points to chunk-sample per item (``None`` = full volume)."""
    num_geometry_points: int | None = None
    """If set, also emit ``geometry_position`` — an independent draw of this many surface points (AB-UPT)."""
    sampling_seed: int | None = None
    """Seed for deterministic chunk selection (``None`` = fresh subset each call)."""
    read_concurrency: int = 1
    """Threads used to fetch a sample's chunks in parallel (``1`` = serial; raise for S3)."""


class ZarrAeroDataset(AeroDataset):
    """AeroDataset reading from a converted Zarr store with chunk-based subsampling."""

    def __init__(
        self,
        dataset_config: StandardDatasetConfig,
        filemap: FileMap,
        num_points: dict[str, int | None] | None = None,
        sampling_seed: int | None = None,
        read_concurrency: int = 1,
        num_geometry_points: int | None = None,
    ) -> None:
        """

        Args:
            dataset_config: Standard dataset config; ``root`` points at the Zarr store root.
            filemap: Field-to-filename mapping (same one used for conversion).
            num_points: Per-domain target counts (``{"surface": 3586, "volume": 4096}``).
                A ``None`` value reads the whole domain. Defaults to full reads.
            sampling_seed: If set, chunk selection is deterministic per sample
                (seed ``sampling_seed + idx``); otherwise a fresh subset is drawn each call.
            read_concurrency: Threads used to fetch a sample's chunks in parallel. Keep at
                ``1`` for local stores; raise it to hide per-request latency on S3.
            num_geometry_points: If set, also emit ``geometry_position`` — an independent
                random draw of this many surface points (the AB-UPT shape-encoder input,
                distinct from the surface anchor points). ``None`` disables it.
        """
        super().__init__(dataset_config=dataset_config, filemap=filemap)
        # Store roots may live on object storage (oci://, s3://, …); only validate local paths.
        root = str(dataset_config.root)
        self.store_root: str | Path = root if stores.is_remote(root) else validate_path(root)
        self.manifest = StoreManifest.load(self.store_root)
        self.reader = ZarrChunkReader(self.store_root, self.manifest, read_concurrency=read_concurrency)
        self.num_points = num_points or {}
        self.sampling_seed = sampling_seed
        self.num_geometry_points = num_geometry_points
        if self.num_geometry_points and _GEOMETRY_SOURCE_DOMAIN not in self.manifest.domains:
            raise ValueError(
                f"geometry_position requires a '{_GEOMETRY_SOURCE_DOMAIN}' domain in the store, "
                f"found {sorted(self.manifest.domains)}."
            )

        self._filename_to_canonical = filename_to_canonical(filemap)
        self._available_fields = self._collect_available_fields()
        # Per-sample read cache so all getitem_* of one sample share a single chunk read.
        # Bounded because PropertySubsetWrapper calls getitem_* directly (bypassing
        # pre_getitem/post_getitem), so we cannot rely on those hooks to clear it.
        self._sample_cache: dict[int, dict[str, torch.Tensor]] = {}
        self._cache_capacity = 8

    def _collect_available_fields(self) -> set[str]:
        """Canonical field names physically present in the store (from the manifest layout)."""
        fields: set[str] = set()
        for layout in self.manifest.domains.values():
            fields.update(layout.arrays)
        return fields

    def get_all_getitem_names(self) -> list[str]:
        """Restrict to ``getitem_*`` for stored fields (+ computed/derived ones that apply)."""
        allowed: list[str] = []
        for name in super().get_all_getitem_names():
            key = name[len("getitem_") :]
            if key in self._available_fields:
                allowed.append(name)
            elif name in _COMPUTED_GETITEMS and "surface_normals" in self._available_fields:
                allowed.append(name)
            elif name == "getitem_geometry_position" and self.num_geometry_points:
                allowed.append(name)
        return allowed

    def _sample_id(self, idx: int) -> str:
        raise NotImplementedError("Subclasses must map an index to a manifest sample id.")

    def _read_sample(self, idx: int) -> dict[str, torch.Tensor]:
        """Chunk-read all fields (and the optional geometry draw) for one sample."""
        generator = None
        if self.sampling_seed is not None:
            generator = torch.Generator().manual_seed(self.sampling_seed + idx)
        sample_id = self._sample_id(idx)
        fields = self.reader.read_sample(sample_id, num_points=self.num_points, generator=generator)
        if self.num_geometry_points:
            # Independent draw of surface positions. Reusing `generator` continues its
            # sequence, so the geometry selection is uncorrelated with the surface-field one.
            fields["geometry_position"] = self.reader.read_coords(
                sample_id, _GEOMETRY_SOURCE_DOMAIN, self.num_geometry_points, generator
            )
        return fields

    def _ensure_loaded(self, idx: int) -> dict[str, torch.Tensor]:
        """Return the cached fields for *idx*, reading them once on first access (bounded cache)."""
        cached = self._sample_cache.get(idx)
        if cached is None:
            if len(self._sample_cache) >= self._cache_capacity:
                self._sample_cache.clear()
            cached = self._read_sample(idx)
            self._sample_cache[idx] = cached
        return cached

    def pre_getitem(self, idx: int) -> dict[str, Any]:
        """Pre-read the sample so all getitem_* share one chunk read (used by ``__getitem__``)."""
        self._ensure_loaded(idx)
        return {}

    @with_normalizers(f"{_GEOMETRY_SOURCE_DOMAIN}_position")
    def getitem_geometry_position(self, idx: int) -> torch.Tensor:
        """Random draw of surface positions used as the AB-UPT geometry/shape-encoder input.

        Independent of the surface anchor points; only emitted when ``num_geometry_points``
        is set. Normalized with the surface position normalizer.
        """
        return self._ensure_loaded(idx)["geometry_position"]

    def post_getitem(self, idx: int, pre: dict[str, Any] | None) -> None:
        """Drop the cached tensors for *idx* (used by ``__getitem__``)."""
        self._sample_cache.pop(idx, None)

    def _load(self, idx: int, filename: str) -> torch.Tensor:
        """Return a (raw, un-normalized) field tensor, reading the sample's chunks on first access."""
        canonical = self._filename_to_canonical[filename]
        return self._ensure_loaded(idx)[canonical]
