Blob-Storage-Efficient Zarr Store
=================================

The default CFD layout stores every field of every sample as a separate ``.pt`` file.
This is simple but ill-suited to blob storage (S3): a sample is many imbalanced
objects, and because training subsamples only a few thousand points per sample, the
whole sample is fetched and then mostly discarded — large *read amplification*.

The :mod:`noether.data.zarr_store` package provides a chunked, sharded, pre-shuffled
`Zarr <https://zarr.dev>`_ alternative that lets the dataloader **read only the points
it samples**.

Format
------

Each sample is an independent Zarr *group* holding **one array per field**
(``surface/position``, ``surface/pressure``, ``volume/velocity``, …): positions are
float32, physical quantities float16. Storing fields separately means any subset of
fields can be read without transferring the rest (see the reader's ``fields=`` argument).

Arrays are chunked along the point axis only and packed into a **single whole-array
shard** by default, so the per-sample object count stays at one object per field while
individual chunks remain range-readable inside the shard. For datasets whose per-field
arrays grow large, cap the shard with ``--shard-points`` (rounded down to a whole number
of chunks; shard bytes ≈ ``shard_points × dim × dtype_size``) — smaller shards bound the
writer's per-shard memory and a corrupt object's blast radius at the cost of more
objects. All arrays of a domain share the chunk and shard grid plus the per-sample
shuffle permutation, so chunk ``c`` addresses the same physical points in every field.
Points are **shuffled once at write time**, so any contiguous chunk is already a
uniform-random subset of the sample.

A small ``manifest.json`` at the store root records the global column layout once and
the per-sample chunk grid (point count, chunk size, shard size, number of chunks).

Why this is fast on S3
----------------------

* **Partial reads.** Reading one chunk from a sharded array is a byte-range request.
  Subsampling ``T`` points reads ``ceil(T / chunk_points)`` chunks per array instead of
  the whole sample — bytes transferred scale with ``T``, not sample size.
* **Random sampling without scatter.** Because points are pre-shuffled, a *random chunk*
  is a *random subset*; epoch-to-epoch diversity comes from picking different chunk
  indices, with no scattered per-point gathers.
* **Smaller + fewer objects.** float16 + zstd roughly halves field bytes, and one shard
  per array keeps object count low while preserving chunk-level read granularity.

Converting a dataset
--------------------

The converter is dataset-agnostic: point it at any FileMap-based ``AeroDataset`` by its
``kind`` and ``root``, and it reuses the dataset's own ``FileMap``, split lists and
per-sample ``_load`` (which encode the on-disk layout) — so ShapeNet-Car, DrivAerML,
AhmedML and DrivAerNet all convert the same way:

.. code-block:: bash

   uv run python -m noether.data.zarr_store.convert \
       --dataset-kind noether.data.datasets.cfd.DrivAerMLDataset \
       --root /path/to/drivaerml \
       --output s3://my-bucket/drivaerml/zarr_store \
       --splits train test val \
       --chunk-points 16384 \
       --workers 16

All requested splits are written into a single store (sample ids are unique across
splits); unsupported splits are skipped with a warning. Pick ``--chunk-points`` close to
the training subsample size to minimise read amplification.

The source can also live on object storage: pass ``--source-url`` instead of ``--root``
(any fsspec location — ``oci://bucket@namespace/prefix``, ``s3://bucket/prefix``, or a
plain directory) and samples are discovered by listing the prefix and streamed without a
local staging copy. The dataset's ``FileMap`` is resolved from ``--dataset-kind``. For
OCI install ``ocifs`` and set ``OCIFS_IAM_TYPE`` (e.g. ``api_key`` to use
``~/.oci/config``); for S3 install ``s3fs``.

.. code-block:: bash

   OCIFS_IAM_TYPE=api_key uv run python -m noether.data.zarr_store.convert \
       --dataset-kind noether.data.datasets.cfd.DrivAerNetDataset \
       --source-url oci://emmi-drivaernet@<namespace>/subsampled_volume10x \
       --output /data/drivaernet/zarr_store \
       --chunk-points 16384 --workers 16

``--output`` likewise accepts a local path or any fsspec URL — the writer, manifest and
reader all resolve store roots through fsspec, so stores can live directly on object
storage. ``--workers`` converts samples in a **process pool** (each worker owns its
source handle and writer), so the GIL-bound ``torch.load``/numpy work parallelizes fully;
the result is bit-identical to a sequential run (shuffles are seeded per sample id).

In code, :func:`noether.data.zarr_store.convert.convert` does the same; for full control
build a :class:`~noether.data.zarr_store.writer.ZarrStoreWriter` and call
:func:`~noether.data.zarr_store.convert.convert_aero_dataset` per dataset/split, or
:func:`~noether.data.zarr_store.convert.convert_fsspec_source` for an fsspec ``.pt`` tree.

Training against the store
--------------------------

:class:`~noether.data.datasets.cfd.ZarrShapeNetCarDataset` reads from the converted store
and is fully config-driven via
:class:`~noether.data.datasets.cfd.ZarrShapeNetCarDatasetConfig` (``kind`` already points at
the dataset, so it resolves through the standard factory). Set ``num_surface_points`` /
``num_volume_points`` to chunk-subsample at read time — the pipeline's
``PointSamplingSampleProcessor`` then becomes a no-op automatically (it skips whenever the
requested count is at least the available points). Leave them ``None`` for full-sample reads
at evaluation. ``num_geometry_points`` additionally emits ``geometry_position`` (an
independent surface draw) for AB-UPT.

.. code-block:: python

   from noether.data.datasets.cfd import ZarrShapeNetCarDataset, ZarrShapeNetCarDatasetConfig

   cfg = ZarrShapeNetCarDatasetConfig(
       root="/path/to/dataset/zarr_store",
       split="train",
       num_surface_points=3586,
       num_volume_points=4096,
       num_geometry_points=3586,  # optional, AB-UPT geometry input
       read_concurrency=1,  # raise (~chunks/sample) to hide latency on S3
   )
   dataset = ZarrShapeNetCarDataset(cfg)

Because the config carries a ``kind`` of ``noether.data.datasets.cfd.ZarrShapeNetCarDataset``,
it can be selected from YAML/Hydra exactly like the other datasets (``kind: ${dataset_kind}``)
with the ``num_*`` fields set on the dataset config.

:class:`~noether.data.datasets.cfd.ZarrDrivAerNetDataset` /
:class:`~noether.data.datasets.cfd.ZarrDrivAerNetDatasetConfig` work the same way for
DrivAerNet(++): ``root`` may be a remote store
(e.g. ``oci://emmi-drivaernet@frwnorq7ern2/zarr_store``), and the split files
(``{train,val,test}_design_ids.txt``) plus blacklists (``blacklist.txt``,
``blacklist2.txt``) are read from the store root via fsspec, so the store is
self-contained. ``filter_categories`` matches the ``.pt`` dataset's behaviour. See
``recipes/aero_cfd/configs/experiment/drivaernet/ab_upt_zarr.yaml`` for an end-to-end
AB-UPT experiment against the OCI store.

Validate a converted store (equivalence vs ``.pt`` and read-amplification) with::

   uv run python -m noether.data.zarr_store.benchmark \
       --pt-root /path/to/dataset --zarr-root /path/to/dataset/zarr_store

Compute normalization statistics (``stats.yaml`` inputs) directly from a store with
:func:`~noether.data.zarr_store.statistics.calculate_store_statistics` — one streaming
pass yields ``{field}_mean/std/min/max``, the logscale moments and the global
``raw_pos_min``/``raw_pos_max`` position bounds::

   uv run python -m noether.data.zarr_store.statistics \
       --store oci://bucket@namespace/zarr_store \
       --split-file train_design_ids.txt \
       --workers 8 --read-concurrency 4 --output-json stats.json

``--split-file`` restricts the pass to the ids listed in a file (a bare name is resolved
against the store root); ids missing from the store are skipped with a warning.

Notes and limitations
---------------------

* float16 fields introduce a small, bounded error (positions stay float32, lossless);
  set ``--values-dtype float32`` if a field needs full precision.
* Pre-shuffling reorders points, so stored connectivity (e.g. ``edge_index``) is **not**
  carried over — the format targets point-cloud sampling, not graph models.
* Store roots may be fsspec URLs: :class:`~noether.data.zarr_store.reader.ZarrChunkReader`
  and the datasets built on it read directly from object storage (raise the dataset's
  ``read_concurrency`` there to hide per-request latency). The ``store_factory`` hook
  remains available for custom backends or instrumentation.
* **Faster S3 backend (optional).** For ``s3://`` roots, installing the optional
  ``obstore`` package (``uv sync --extra obstore``) makes
  :func:`~noether.data.zarr_store.stores.make_store` transparently use the Rust-backed
  :class:`zarr.storage.ObjectStore` instead of :class:`~zarr.storage.FsspecStore`;
  credentials/region/endpoint come from the standard ``AWS_*`` environment variables.
  It is a drop-in speedup (≈2× on the warm per-sample read path in benchmarks) and falls
  back to fsspec automatically when obstore is absent or the URL is not ``s3://``. OCI is
  reachable this way via its S3-compatible endpoint (``AWS_ENDPOINT_URL`` +
  ``s3://bucket/...``) using a Customer Secret Key, but the default ``oci://`` path keeps
  using ``ocifs``/API-key auth.
