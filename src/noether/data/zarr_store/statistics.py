#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Calculate per-field statistics of a converted Zarr store.

The Zarr-store counterpart of :mod:`noether.data.tools.calculate_statistics`: instead of
going through a dataset class, it streams every sample's fields straight from the store
(local path or fsspec URL such as ``oci://bucket@namespace/zarr_store``) and accumulates
running moments in a single pass. Linear and logscale moments are computed together via
:class:`~noether.data.stats.RunningStats`, so one run yields everything a ``stats.yaml``
needs (``{field}_mean/std/min/max`` plus ``{field}_logscale_mean/std`` and the global
``raw_pos_min``/``raw_pos_max`` position bounds).

Usage::

    OCIFS_IAM_TYPE=api_key uv run python -m noether.data.zarr_store.statistics \\
        --store oci://emmi-drivaernet@frwnorq7ern2/zarr_store \\
        --split-file train_design_ids.txt \\
        --workers 8 --read-concurrency 4 --output-json drivaernet_stats.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
import torch
from tqdm import tqdm

from noether.data.stats import RunningStats
from noether.data.zarr_store import stores
from noether.data.zarr_store.manifest import StoreManifest
from noether.data.zarr_store.reader import ZarrChunkReader


def _store_fields(manifest: StoreManifest) -> set[str]:
    """All canonical field names physically present in the store."""
    fields: set[str] = set()
    for layout in manifest.domains.values():
        fields.update(layout.arrays)
    return fields


def _resolve_fields(
    manifest: StoreManifest,
    fields: set[str] | None,
    exclude_fields: set[str] | None,
) -> list[str]:
    """Validate the include/exclude selection against the store's fields."""
    available = _store_fields(manifest)
    selected = set(fields) if fields else set(available)
    unknown = selected - available
    if unknown:
        raise ValueError(f"Unknown fields {sorted(unknown)}; store has {sorted(available)}.")
    if exclude_fields:
        unknown = exclude_fields - available
        if unknown:
            raise ValueError(f"Cannot exclude non-existent fields {sorted(unknown)}.")
        selected -= exclude_fields
    return sorted(selected)


def read_split_ids(store_root: str | Path, split_file: str) -> list[str]:
    """Read sample ids (one per line) from a split file.

    ``split_file`` may be an absolute path/URL or a name relative to the store root
    (e.g. ``train_design_ids.txt`` next to ``manifest.json``).
    """
    url = split_file if "://" in split_file or Path(split_file).is_absolute() else stores.join(store_root, split_file)
    with fsspec.open(url, "r") as f:
        return [line.strip() for line in f if line.strip()]


def calculate_store_statistics(
    store_root: str | Path,
    *,
    fields: set[str] | None = None,
    exclude_fields: set[str] | None = None,
    sample_ids: list[str] | None = None,
    limit: int | None = None,
    max_workers: int = 1,
    read_concurrency: int = 1,
    progress: bool = False,
) -> dict[str, RunningStats]:
    """Stream all samples of a Zarr store and accumulate per-field running statistics.

    Args:
        store_root: Store root (local path or fsspec URL).
        fields: Restrict to these canonical field names (default: every stored field).
        exclude_fields: Field names to skip.
        sample_ids: Restrict to these manifest sample ids (default: all samples).
        limit: Process at most this many samples (after ``sample_ids`` filtering).
        max_workers: Samples read concurrently (threads); accumulation stays single-threaded.
        read_concurrency: Per-sample chunk-read threads (see :class:`ZarrChunkReader`).
        progress: Show a tqdm progress bar.

    Returns:
        Mapping from canonical field name to its :class:`~noether.data.stats.RunningStats`
        (per-component mean/std/min/max and logscale moments, accumulated in float64).

    Raises:
        ValueError: If a requested field is not present in the store, or no samples remain
            to process. Sample ids missing from the store are skipped with a warning
            (split files may list samples that were skipped at conversion).
    """
    manifest = StoreManifest.load(store_root)
    reader = ZarrChunkReader(store_root, manifest, read_concurrency=read_concurrency)
    selected = _resolve_fields(manifest, fields, exclude_fields)

    ids = sample_ids if sample_ids is not None else sorted(manifest.samples)
    missing = [sid for sid in ids if sid not in manifest.samples]
    if missing:
        # Split files may list samples that were skipped at conversion (e.g. blacklisted).
        warnings.warn(f"Skipping {len(missing)} sample ids not in the store (e.g. {missing[0]}).", stacklevel=2)
        ids = [sid for sid in ids if sid in manifest.samples]
    ids = ids[:limit]
    if not ids:
        raise ValueError("No samples to process.")

    running_stats = {field: RunningStats(name=field) for field in selected}

    def _read(sid: str) -> dict[str, torch.Tensor]:
        return reader.read_sample(sid, fields=set(selected))

    # Threads fetch full samples; pushing stays in the main thread so the Welford
    # accumulators need no locking (push order does not change the result materially).
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = pool.map(_read, ids)
        for sample in tqdm(results, total=len(ids), desc="Processing store", disable=not progress):
            for field, value in sample.items():
                running_stats[field].push_tensor(value)

    return running_stats


def _to_list(x: torch.Tensor) -> list[float]:
    return [float(v) for v in x.tolist()]


def statistics_to_dict(running_stats: dict[str, RunningStats]) -> dict[str, list[float] | int]:
    """Flatten running statistics into ``stats.yaml``-style keys.

    Emits ``{field}_mean/std/min/max/count`` and ``{field}_logscale_mean/std`` per field,
    plus global ``raw_pos_min``/``raw_pos_max`` scalars over all ``*_position`` fields
    (the bounds used by position normalization).
    """
    out: dict[str, list[float] | int] = {}
    pos_min: float | None = None
    pos_max: float | None = None
    for field, stats in sorted(running_stats.items()):
        out[f"{field}_mean"] = _to_list(stats.mean)
        out[f"{field}_std"] = _to_list(stats.std)
        out[f"{field}_min"] = _to_list(stats.min)
        out[f"{field}_max"] = _to_list(stats.max)
        out[f"{field}_logscale_mean"] = _to_list(stats.logmean)
        out[f"{field}_logscale_std"] = _to_list(stats.logstd)
        out[f"{field}_count"] = stats.count
        if field.endswith("_position"):
            lo, hi = float(stats.min.min()), float(stats.max.max())
            pos_min = lo if pos_min is None else min(pos_min, lo)
            pos_max = hi if pos_max is None else max(pos_max, hi)
    if pos_min is not None and pos_max is not None:
        out["raw_pos_min"] = [pos_min]
        out["raw_pos_max"] = [pos_max]
    return out


def print_statistics(running_stats: dict[str, RunningStats]) -> None:
    """Print the accumulated statistics per field, plus the global position bounds."""
    for _, stats in sorted(running_stats.items()):
        print(stats)
    flat = statistics_to_dict(running_stats)
    if "raw_pos_min" in flat:
        print(f"raw_pos_min: {flat['raw_pos_min']}")
        print(f"raw_pos_max: {flat['raw_pos_max']}")


def save_statistics_to_json(running_stats: dict[str, RunningStats], output_path: str | Path) -> None:
    """Save the flattened statistics (see :func:`statistics_to_dict`) as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(statistics_to_dict(running_stats), f, indent=2)
    print(f"\nStatistics saved to: {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Calculate per-field statistics of a converted Zarr store.")
    parser.add_argument("--store", required=True, help="Store root (local path or fsspec URL, e.g. oci://...).")
    parser.add_argument(
        "--fields",
        type=lambda s: set(s.split(",")) if s else None,
        default=None,
        help="Comma-separated canonical field names to include (default: all stored fields).",
    )
    parser.add_argument(
        "--exclude-fields",
        type=lambda s: set(s.split(",")) if s else None,
        default=None,
        help="Comma-separated canonical field names to exclude.",
    )
    parser.add_argument(
        "--split-file",
        default=None,
        help="Optional file with one sample id per line; a bare name is resolved relative to "
        "the store root (e.g. train_design_ids.txt). Default: all samples in the manifest.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many samples.")
    parser.add_argument("--workers", type=int, default=1, help="Samples read concurrently.")
    parser.add_argument("--read-concurrency", type=int, default=1, help="Chunk-read threads per sample.")
    parser.add_argument("--output-json", default=None, help="Optional path to save the statistics as JSON.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point for calculating Zarr store statistics."""
    args = _parse_args()
    sample_ids = read_split_ids(args.store, args.split_file) if args.split_file else None
    running_stats = calculate_store_statistics(
        args.store,
        fields=args.fields,
        exclude_fields=args.exclude_fields,
        sample_ids=sample_ids,
        limit=args.limit,
        max_workers=args.workers,
        read_concurrency=args.read_concurrency,
        progress=True,
    )
    print_statistics(running_stats)
    if args.output_json:
        save_statistics_to_json(running_stats, args.output_json)


if __name__ == "__main__":
    main()
