#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Convert a per-field ``.pt`` CFD dataset into the chunked/sharded Zarr store.

Works for any FileMap-based :class:`~noether.data.datasets.cfd.dataset.AeroDataset`
(ShapeNet-Car, DrivAerML, AhmedML, DrivAerNet, …) from two kinds of sources:

* ``--root`` — a local/NFS dataset root: the dataset class itself supplies the
  ``FileMap``, the per-split sample lists and the directory layout (via ``_load``).
* ``--source-url`` — any fsspec location (``oci://bucket@namespace/prefix``,
  ``s3://bucket/prefix``, a plain directory, …): samples are discovered by listing the
  prefix and grouping ``.pt`` files by directory, then streamed without a local copy.

Samples are converted in a :class:`~concurrent.futures.ProcessPoolExecutor` so the
GIL-bound parts (``torch.load`` deserialization, numpy shuffles) parallelize fully;
each worker process builds its own source handle and writer, and the manifest is
assembled in the parent from the returned entries (bit-identical to a sequential run —
per-sample shuffles are seeded by sample id).

Usage::

    uv run python -m noether.data.zarr_store.convert \
        --dataset-kind noether.data.datasets.cfd.DrivAerNetDataset \
        --source-url oci://emmi-drivaernet@<namespace>/subsampled_volume10x \
        --output /data/drivaernet/zarr_store \
        --chunk-points 16384 --workers 16
"""

from __future__ import annotations

import argparse
import inspect
import io
import logging
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Any

import fsspec
import numpy as np
import torch
from tqdm import tqdm

from noether.core.factory.utils import class_constructor_from_class_path
from noether.core.schemas.lib import resolve_config_class
from noether.data.base.dataset import DatasetBaseConfig
from noether.data.schemas import FileMap
from noether.data.zarr_store.layout import filename_to_canonical, present_specs
from noether.data.zarr_store.writer import ZarrStoreWriter

if TYPE_CHECKING:
    from noether.data.datasets.cfd.dataset import AeroDataset
    from noether.data.zarr_store.manifest import SampleEntry

logger = logging.getLogger(__name__)

# One fsspec conversion task: (sample_id, canonical_field -> filesystem path).
FsspecTask = tuple[str, dict[str, str]]


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #
def build_dataset(kind: str, root: str, split: str) -> AeroDataset:
    """Instantiate a dataset ``kind`` for one split (raw: no normalizers/wrappers)."""
    dataset_cls = class_constructor_from_class_path(kind)
    # Concrete config (StandardDatasetConfig subclass) carries root/split; typed Any so the
    # call isn't checked against the abstract DatasetBaseConfig signature.
    config_cls: Any = resolve_config_class(kind, DatasetBaseConfig)
    config = config_cls(kind=kind, root=root, split=split)
    return dataset_cls(config)  # type: ignore[no-any-return]


def filemap_for_dataset_kind(kind: str) -> FileMap:
    """Resolve a dataset kind's ``FileMap`` without instantiating the dataset.

    Looks for, in order: a ``FILEMAP`` class attribute, a ``filemap`` ``__init__``
    default anywhere in the MRO, and finally a ``FileMap`` instance in the dataset's
    module globals.
    """
    dataset_cls: Any = class_constructor_from_class_path(kind)
    candidate = getattr(dataset_cls, "FILEMAP", None)
    if isinstance(candidate, FileMap):
        return candidate
    for klass in dataset_cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        try:
            parameter = inspect.signature(init).parameters.get("filemap")
        except (TypeError, ValueError):
            continue
        if parameter is not None and isinstance(parameter.default, FileMap):
            return parameter.default
    module = sys.modules.get(dataset_cls.__module__)
    for value in vars(module).values() if module else ():
        if isinstance(value, FileMap):
            return value
    raise ValueError(f"Could not resolve a FileMap for '{kind}'; pass one explicitly.")


def discover_fsspec_samples(source_url: str, filemap: FileMap, limit: int | None = None) -> list[FsspecTask]:
    """List *source_url* and group its ``.pt`` files into per-sample conversion tasks.

    A sample is any directory (prefix) containing every field of *filemap*; incomplete
    samples are skipped with a warning. Sample ids are the directory paths relative to
    the prefix (e.g. ``"E_S_WWC_WM_005"`` or ``"param1/<hash>"``), matching the ids the
    dataset-driven path produces.
    """
    fs, base = fsspec.url_to_fs(str(source_url))
    base = base.rstrip("/")
    to_canonical = filename_to_canonical(filemap)
    expected = {spec.canonical for spec in present_specs(filemap)}

    grouped: dict[str, dict[str, str]] = defaultdict(dict)
    for path in fs.find(base):
        relative = path[len(base) :].lstrip("/")
        sample_id, _, filename = relative.rpartition("/")
        canonical = to_canonical.get(filename)
        if sample_id and canonical:
            grouped[sample_id][canonical] = path

    tasks = [(sample_id, fields) for sample_id, fields in sorted(grouped.items()) if set(fields) == expected]
    if len(tasks) < len(grouped):
        logger.warning("Skipping %d samples with missing fields under %s", len(grouped) - len(tasks), source_url)
    return tasks[:limit] if limit is not None else tasks


def _sample_id(dataset: AeroDataset, idx: int) -> str:
    """Stable manifest id for sample *idx* — the dataset's run/design id, else its index."""
    sample_info = getattr(dataset, "sample_info", None)
    if callable(sample_info):
        info = sample_info(idx) or {}
        for key in ("run_name", "design_id"):
            value = info.get(key)
            if value is not None:
                return str(value)
    design_ids = getattr(dataset, "design_ids", None)
    if design_ids is not None:
        return str(design_ids[idx])
    return str(idx)


def _load_dataset_fields(dataset: AeroDataset, idx: int) -> dict[str, np.ndarray]:
    """Read all present fields of sample *idx* through the dataset's own ``_load`` (raw)."""
    filemap = dataset.filemap
    return {
        spec.canonical: np.asarray(dataset._load(idx, getattr(filemap, spec.filemap_attr)).numpy())
        for spec in present_specs(filemap)
    }


# --------------------------------------------------------------------------- #
# process-pool machinery
# --------------------------------------------------------------------------- #
# Per-process state, populated by the pool initializer (each worker owns its source
# handle and writer; only the small SampleEntry results travel back to the parent).
_WORKER: dict[str, Any] = {}


def _init_dataset_worker(dataset: AeroDataset, writer_kwargs: dict[str, Any]) -> None:
    _WORKER["dataset"] = dataset
    _WORKER["writer"] = ZarrStoreWriter(**writer_kwargs)


def _convert_dataset_sample(idx: int) -> tuple[str, SampleEntry | None, str | None]:
    dataset, writer = _WORKER["dataset"], _WORKER["writer"]
    sample_id = _sample_id(dataset, idx)
    try:
        return sample_id, writer.write_group(sample_id, _load_dataset_fields(dataset, idx)), None
    except Exception as exc:  # broken sample (e.g. misaligned fields) — parent reports and skips it
        return sample_id, None, f"{type(exc).__name__}: {exc}"


def _init_fsspec_worker(source_url: str, writer_kwargs: dict[str, Any]) -> None:
    _WORKER["fs"], _ = fsspec.url_to_fs(source_url)
    _WORKER["writer"] = ZarrStoreWriter(**writer_kwargs)


def _convert_fsspec_sample(task: FsspecTask) -> tuple[str, SampleEntry | None, str | None]:
    sample_id, field_paths = task
    fs, writer = _WORKER["fs"], _WORKER["writer"]
    try:
        field_arrays = {
            canonical: np.asarray(torch.load(io.BytesIO(fs.cat_file(path)), weights_only=True).numpy())
            for canonical, path in field_paths.items()
        }
        return sample_id, writer.write_group(sample_id, field_arrays), None
    except Exception as exc:  # broken sample (e.g. misaligned fields) — parent reports and skips it
        return sample_id, None, f"{type(exc).__name__}: {exc}"


def _run_pool(
    tasks: Iterable[Any],
    task_fn: Callable[[Any], tuple[str, SampleEntry | None, str | None]],
    initializer: Callable[..., None],
    initargs: tuple[Any, ...],
    writer: ZarrStoreWriter,
    max_workers: int,
    desc: str,
) -> list[tuple[str, str]]:
    """Convert *tasks* (in worker processes when ``max_workers > 1``) and collect entries.

    Broken samples (e.g. misaligned fields) are skipped with a warning instead of aborting
    the run; the ``(sample_id, error)`` pairs of skipped samples are returned. Raises only
    when *every* task failed (a systemic problem, not a data quirk).
    """
    tasks = list(tasks)
    failures: list[tuple[str, str]] = []

    def _collect(result: tuple[str, SampleEntry | None, str | None]) -> None:
        sample_id, entry, error = result
        if entry is None:
            failures.append((sample_id, error or "unknown error"))
            logger.warning("Skipping sample '%s': %s", sample_id, error)
        else:
            writer.manifest.samples[sample_id] = entry  # parent process only
        bar.update()

    bar = tqdm(total=len(tasks), desc=desc, unit="sample")
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers, initializer=initializer, initargs=initargs) as executor:
            for result in executor.map(task_fn, tasks):
                _collect(result)
    else:
        initializer(*initargs)
        try:
            for task in tasks:
                _collect(task_fn(task))
        finally:
            _WORKER.clear()
    bar.close()

    if failures:
        skipped = ", ".join(sample_id for sample_id, _ in failures[:5])
        logger.warning("Skipped %d/%d samples due to errors (e.g. %s)", len(failures), len(tasks), skipped)
    if tasks and len(failures) == len(tasks):
        raise RuntimeError(f"All {len(tasks)} samples failed; first error ({failures[0][0]}): {failures[0][1]}")
    return failures


# --------------------------------------------------------------------------- #
# converters
# --------------------------------------------------------------------------- #
def convert_aero_dataset(
    dataset: AeroDataset, writer: ZarrStoreWriter, *, max_workers: int = 1, limit: int | None = None
) -> ZarrStoreWriter:
    """Write every sample of a FileMap-based dataset into *writer* (does not save the manifest).

    Args:
        dataset: A FileMap-based ``AeroDataset`` instance (one split).
        writer: Target store writer; reused across splits to build one store.
        max_workers: Worker *processes* converting samples concurrently; the dataset and
            a writer clone are set up once per process.
        limit: If set, convert at most this many samples (debugging).

    Returns:
        The same ``writer`` (call :meth:`ZarrStoreWriter.save_manifest` once when done).

    Raises:
        ValueError: If the dataset does not expose a ``filemap``.
    """
    if not isinstance(getattr(dataset, "filemap", None), FileMap):
        raise ValueError(f"{type(dataset).__name__} has no FileMap; the Zarr converter requires one.")
    count = len(dataset) if limit is None else min(limit, len(dataset))
    desc = f"split={getattr(dataset, 'split', '?')}"
    _run_pool(
        range(count),
        _convert_dataset_sample,
        _init_dataset_worker,
        (dataset, writer.to_init_kwargs()),
        writer,
        max_workers,
        desc,
    )
    return writer


def convert_fsspec_source(
    source_url: str, filemap: FileMap, writer: ZarrStoreWriter, *, max_workers: int = 1, limit: int | None = None
) -> ZarrStoreWriter:
    """Stream-convert a remote (or local) ``.pt`` tree reachable through fsspec.

    Samples are discovered once in the parent by listing *source_url*; each worker
    process opens its own filesystem handle and reads sample files directly from the
    source (no local staging copy).

    Args:
        source_url: fsspec location of the ``.pt`` tree (``oci://``, ``s3://``, a path, …).
        filemap: Field-to-filename mapping of the dataset (see :func:`filemap_for_dataset_kind`).
        writer: Target store writer.
        max_workers: Worker processes converting samples concurrently.
        limit: If set, convert at most this many samples (debugging/benchmarks).

    Returns:
        The same ``writer`` (call :meth:`ZarrStoreWriter.save_manifest` once when done).

    Raises:
        RuntimeError: If no complete samples are found under *source_url*.
    """
    tasks = discover_fsspec_samples(source_url, filemap, limit)
    if not tasks:
        raise RuntimeError(f"No complete samples found under {source_url}.")
    logger.info("Discovered %d samples under %s", len(tasks), source_url)
    _run_pool(
        tasks,
        _convert_fsspec_sample,
        _init_fsspec_worker,
        (str(source_url), writer.to_init_kwargs()),
        writer,
        max_workers,
        desc="fsspec-source",
    )
    return writer


def convert(
    dataset_kind: str,
    store_root: str,
    *,
    root: str | None = None,
    source_url: str | None = None,
    splits: list[str] | None = None,
    dataset_name: str | None = None,
    chunk_points: int = 4096,
    shard_points: int | None = None,
    shuffle_seed: int = 0,
    values_dtype: str = "float16",
    field_dtypes: dict[str, str] | None = None,
    limit: int | None = None,
    max_workers: int = 1,
) -> ZarrStoreWriter:
    """Convert a FileMap dataset into a single Zarr store.

    Exactly one of *root* (local dataset-driven, per split) or *source_url* (fsspec
    streaming, all samples under the prefix) must be given.

    Args:
        dataset_kind: Fully-qualified dataset class path (supplies the ``FileMap``).
        store_root: Output Zarr store — a local path or fsspec URL (``s3://``, ``gs://``, …).
        root: Local dataset root (dataset-driven source).
        source_url: fsspec source prefix (streaming source; ``splits`` is ignored).
        splits: Splits to convert for the dataset-driven source (default train/test/val);
            unsupported/empty ones are skipped with a warning.
        dataset_name: Manifest name (defaults to the dataset class name).
        chunk_points: Chunk size along the point axis (pick close to the train subsample size).
        shard_points: Cap on the shard size along the point axis (``None`` = one shard per array).
        shuffle_seed: Base seed for the per-sample point shuffle.
        values_dtype: Dtype for the physical field arrays.
        field_dtypes: Per-field dtype overrides, e.g. ``{"volume_vorticity": "float32"}``
            for fields whose values exceed the float16 range.
        limit: If set, convert at most this many samples per split / source.
        max_workers: Worker processes converting samples concurrently.

    Returns:
        The writer with the saved manifest.

    Raises:
        ValueError: If not exactly one of *root* / *source_url* is given.
        RuntimeError: If nothing could be converted.
    """
    if (root is None) == (source_url is None):
        raise ValueError("Provide exactly one of `root` (local dataset) or `source_url` (fsspec source).")
    name = dataset_name or dataset_kind.rsplit(".", 1)[-1]

    if source_url is not None:
        filemap = filemap_for_dataset_kind(dataset_kind)
        writer = ZarrStoreWriter(
            store_root=store_root,
            filemap=filemap,
            dataset_name=name,
            shuffle_seed=shuffle_seed,
            chunk_points=chunk_points,
            shard_points=shard_points,
            values_dtype=values_dtype,
            field_dtypes=field_dtypes,
        )
        convert_fsspec_source(source_url, filemap, writer, max_workers=max_workers, limit=limit)
        writer.save_manifest()
        logger.info("Done. %d samples from %s -> %s", len(writer.manifest.samples), source_url, store_root)
        return writer

    maybe_writer: ZarrStoreWriter | None = None
    total = 0
    for split in splits or ["train", "test", "val"]:
        try:
            dataset = build_dataset(dataset_kind, str(root), split)
        except Exception as exc:  # unsupported split, missing data, bad root, …
            logger.warning("Skipping split '%s': %s", split, exc)
            continue
        if maybe_writer is None:
            maybe_writer = ZarrStoreWriter(
                store_root=store_root,
                filemap=dataset.filemap,
                dataset_name=name,
                shuffle_seed=shuffle_seed,
                chunk_points=chunk_points,
                shard_points=shard_points,
                values_dtype=values_dtype,
                field_dtypes=field_dtypes,
            )
        convert_aero_dataset(dataset, maybe_writer, max_workers=max_workers, limit=limit)
        converted = len(dataset) if limit is None else min(limit, len(dataset))
        total += converted
        logger.info("Converted split '%s' (%d samples)", split, converted)

    if maybe_writer is None:
        raise RuntimeError(f"No splits converted for {dataset_kind} (tried {splits}). Check --root/--splits.")
    maybe_writer.save_manifest()
    logger.info("Done. %d samples -> %s", total, store_root)
    return maybe_writer


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a FileMap .pt CFD dataset into a chunked/sharded Zarr store.")
    parser.add_argument(
        "--dataset-kind",
        required=True,
        help="Dataset class path, e.g. noether.data.datasets.cfd.DrivAerNetDataset.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", help="Local dataset root (dataset-driven source, converted per split).")
    source.add_argument(
        "--source-url",
        help="fsspec source prefix to stream from, e.g. oci://bucket@namespace/prefix or s3://bucket/prefix.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output Zarr store root: a local path or an fsspec URL (e.g. s3://bucket/prefix).",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "test", "val"], help="Splits (local source only).")
    parser.add_argument("--workers", type=int, default=1, help="Worker processes converting samples concurrently.")
    parser.add_argument("--chunk-points", type=int, default=4096, help="Chunk size along the point axis.")
    parser.add_argument(
        "--shard-points",
        type=int,
        default=None,
        help="Cap on the shard size along the point axis (default: one shard per array).",
    )
    parser.add_argument("--shuffle-seed", type=int, default=0, help="Base seed for the point shuffle.")
    parser.add_argument("--values-dtype", type=str, default="float16", help="Dtype for physical fields.")
    parser.add_argument(
        "--field-dtype",
        action="append",
        default=None,
        metavar="FIELD=DTYPE",
        help="Per-field dtype override (repeatable), e.g. --field-dtype volume_vorticity=float32.",
    )
    parser.add_argument("--dataset-name", type=str, default=None, help="Manifest dataset name (default: class name).")
    parser.add_argument("--limit", type=int, default=None, help="Convert at most N samples per split/source (debug).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    field_dtypes = dict(item.split("=", 1) for item in args.field_dtype) if args.field_dtype else None
    convert(
        dataset_kind=args.dataset_kind,
        store_root=args.output,
        root=args.root,
        source_url=args.source_url,
        splits=args.splits,
        dataset_name=args.dataset_name,
        chunk_points=args.chunk_points,
        shard_points=args.shard_points,
        shuffle_seed=args.shuffle_seed,
        values_dtype=args.values_dtype,
        field_dtypes=field_dtypes,
        limit=args.limit,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
