#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Validate and benchmark a converted ShapeNet-Car Zarr store against the ``.pt`` source.

Checks, for the ``test`` split:

1. **Equivalence** — normalized per-field output of :class:`ZarrShapeNetCarDataset`
   (full read) matches the original :class:`ShapeNetCarDataset` within float16 error.
2. **Read amplification** — bytes fetched for a chunk-subsampled read vs a full read,
   measured by instrumenting the Zarr store.

Run::

    uv run python -m noether.data.zarr_store.benchmark \
        --pt-root /nfs-gpu/research/datasets/shapenet_car \
        --zarr-root /nfs-gpu/research/datasets/shapenet_car/zarr_store
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from zarr.storage import LocalStore

from noether.data.base.dataset import StandardDatasetConfig
from noether.data.datasets.cfd.shapenet_car.dataset import ShapeNetCarDataset
from noether.data.datasets.cfd.shapenet_car.zarr_dataset import ZarrShapeNetCarDataset, ZarrShapeNetCarDatasetConfig
from noether.data.preprocessors.normalizers import FieldNormalizerConfig
from noether.data.zarr_store.reader import ZarrChunkReader

FIELDS = [
    "surface_position",
    "surface_pressure",
    "surface_normals",
    "volume_position",
    "volume_velocity",
    "volume_sdf",
    "volume_normals",
]

_POS = FieldNormalizerConfig(strategy="position", scale=1000, stat_keys={"min": "raw_pos_min", "max": "raw_pos_max"})
NORMALIZERS = {
    "surface_pressure": FieldNormalizerConfig(strategy="mean_std"),
    "volume_velocity": FieldNormalizerConfig(strategy="mean_std"),
    "volume_sdf": FieldNormalizerConfig(strategy="mean_std"),
    "surface_position": _POS,
    "volume_position": _POS,
}


def check_equivalence(pt_root: str, zarr_root: str, num_samples: int) -> None:
    """Compare normalized fields of the original and Zarr datasets (full read)."""
    pt_cfg = StandardDatasetConfig(root=pt_root, split="test", dataset_normalizers=NORMALIZERS)
    zarr_cfg = ZarrShapeNetCarDatasetConfig(root=zarr_root, split="test", dataset_normalizers=NORMALIZERS)
    original = ShapeNetCarDataset(pt_cfg)
    converted = ZarrShapeNetCarDataset(zarr_cfg)  # full read (num_* default to None)

    print(f"\n== Equivalence (normalized, {num_samples} samples, sorted per-field) ==")
    worst = dict.fromkeys(FIELDS, 0.0)
    for idx in range(min(num_samples, len(converted))):
        z_sample = converted[idx]
        for field in FIELDS:
            ref = getattr(original, f"getitem_{field}")(idx).numpy()
            got = z_sample[field].numpy()
            ref_s = np.sort(ref.reshape(len(ref), -1), axis=0)
            got_s = np.sort(got.reshape(len(got), -1), axis=0)
            worst[field] = max(worst[field], float(np.abs(ref_s - got_s).max()))
    for field in FIELDS:
        kind = "f32 exact" if "position" in field else "f16"
        print(f"  {field:18s} max abs err = {worst[field]:.2e}   ({kind})")


class _CountingLocalStore(LocalStore):
    """LocalStore that tallies bytes returned for chunk-data keys."""

    bytes_read = 0

    async def get(self, key, prototype, byte_range=None):  # type: ignore[override]
        res = await super().get(key, prototype, byte_range)
        if res is not None and "/c/" in str(key):
            type(self).bytes_read += len(res)
        return res


def check_read_amplification(zarr_root: str, num_volume_points: int) -> None:
    """Measure bytes fetched for a chunk-subsampled read vs a full read."""
    reader = ZarrChunkReader(zarr_root, store_factory=lambda p: _CountingLocalStore(p, read_only=True))
    sample_id = next(iter(reader.manifest.samples))

    _CountingLocalStore.bytes_read = 0
    reader.read_sample(sample_id, num_points={"surface": None, "volume": num_volume_points})
    sub = _CountingLocalStore.bytes_read

    reader._groups.clear()
    _CountingLocalStore.bytes_read = 0
    reader.read_sample(sample_id, num_points=None)
    full = _CountingLocalStore.bytes_read

    print(f"\n== Read amplification (sample {sample_id}) ==")
    print(f"  full read           : {full / 1024:8.1f} KiB")
    print(f"  volume={num_volume_points} subsample: {sub / 1024:8.1f} KiB")
    print(f"  bytes saved         : {100 * (1 - sub / full):.1f}%  (full/sub = {full / sub:.1f}x)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate/benchmark a ShapeNet-Car Zarr store.")
    parser.add_argument("--pt-root", type=str, required=True, help="Original dataset root (contains 'preprocessed').")
    parser.add_argument("--zarr-root", type=str, required=True, help="Converted Zarr store root.")
    parser.add_argument("--num-samples", type=int, default=10, help="Samples to check for equivalence.")
    parser.add_argument("--num-volume-points", type=int, default=4096, help="Volume subsample size for the read test.")
    args = parser.parse_args()

    print(f"Store size: {_dir_size(args.zarr_root) / 1e9:.3f} GB")
    check_equivalence(args.pt_root, args.zarr_root, args.num_samples)
    check_read_amplification(args.zarr_root, args.num_volume_points)


def _dir_size(path: str) -> int:
    total = 0
    for root, _, files in Path(path).walk():
        total += sum((root / f).stat().st_size for f in files)
    return total


if __name__ == "__main__":
    main()
