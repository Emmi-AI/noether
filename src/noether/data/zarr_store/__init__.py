#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Chunked, sharded, pre-shuffled Zarr store for CFD datasets.

This package provides a blob-storage-friendly alternative to the per-field ``.pt``
layout: each sample is a Zarr group with fused float32 ``coords`` and float16
``fields`` arrays per domain, chunked along the point axis and packed into shards.
Pre-shuffling points at write time lets the dataloader subsample by reading a few
random chunks (byte-range reads) instead of loading the whole sample.
"""

from noether.data.zarr_store.layout import (
    CANONICAL_TO_SPEC,
    FIELD_SPECS,
    FieldSpec,
    build_domain_layouts,
    filename_to_canonical,
    present_specs,
)
from noether.data.zarr_store.manifest import (
    ArrayLayout,
    DomainLayout,
    DomainSample,
    SampleEntry,
    StoreManifest,
)
from noether.data.zarr_store.reader import ZarrChunkReader
from noether.data.zarr_store.stores import is_remote, make_store
from noether.data.zarr_store.writer import ZarrStoreWriter

__all__ = [
    "ArrayLayout",
    "CANONICAL_TO_SPEC",
    "DomainLayout",
    "DomainSample",
    "FIELD_SPECS",
    "FieldSpec",
    "SampleEntry",
    "StoreManifest",
    "ZarrChunkReader",
    "ZarrStoreWriter",
    "build_domain_layouts",
    "filename_to_canonical",
    "is_remote",
    "make_store",
    "present_specs",
]
