#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Zarr-backed DrivAerNet(++) dataset with chunk-based subsampling.

Reads from a converted Zarr store (local path or fsspec URL such as
``oci://bucket@namespace/zarr_store``). The store is self-contained: the split files
(``{train,test,val}_design_ids.txt``) and blacklists (``blacklist.txt``,
``blacklist2.txt``) live next to ``manifest.json`` and are read through fsspec, so the
same split/blacklist semantics as :class:`DrivAerNetDataset` apply.
"""

from __future__ import annotations

import logging

import fsspec

from noether.data.datasets.cfd.drivaernet.dataset import VALID_CATEGORIES, DrivAerNetDataset
from noether.data.datasets.cfd.zarr_aero_dataset import ZarrAeroDataset, ZarrAeroDatasetConfig
from noether.data.zarr_store import stores

logger = logging.getLogger(__name__)


class ZarrDrivAerNetDatasetConfig(ZarrAeroDatasetConfig):
    """Config for the Zarr-backed DrivAerNet dataset (``root`` = converted Zarr store)."""

    kind: str | None = "noether.data.datasets.cfd.ZarrDrivAerNetDataset"

    filter_categories: list[str] | None = None
    """Optional design-category filter (e.g. ``["E_S_WWC_WM"]``), as in the ``.pt`` dataset."""


class ZarrDrivAerNetDataset(ZarrAeroDataset):
    """DrivAerNet(++) dataset reading from a converted Zarr store.

    Mirrors :class:`DrivAerNetDataset`'s split handling (split id files, blacklists,
    category filtering and ``"train[0:100]"`` subset notation) while subsampling via
    chunked Zarr reads. When the config's ``num_*`` counts are set, the pipeline's
    anchor/geometry sampling becomes inert (see ``geometry_position_from_dataset``).
    """

    STATS_FILE: str = DrivAerNetDataset.STATS_FILE

    def __init__(self, dataset_config: ZarrDrivAerNetDatasetConfig) -> None:
        """

        Args:
            dataset_config: DrivAerNet Zarr config; ``root`` is the Zarr store root (local
                or fsspec URL) and the ``num_*`` / ``sampling_seed`` / ``read_concurrency``
                fields drive subsampling.
        """
        super().__init__(
            dataset_config=dataset_config,
            filemap=DrivAerNetDataset.FILEMAP,
            num_points={
                "surface": dataset_config.num_surface_points,
                "volume": dataset_config.num_volume_points,
            },
            sampling_seed=dataset_config.sampling_seed,
            read_concurrency=dataset_config.read_concurrency,
            num_geometry_points=dataset_config.num_geometry_points,
        )
        datasplits = self._load_datasplits()
        if dataset_config.filter_categories:
            for category in dataset_config.filter_categories:
                if category not in VALID_CATEGORIES:
                    raise ValueError(f"Invalid category: {category}. Valid categories: {VALID_CATEGORIES}")
            datasplits = {
                split: [i for i in ids if "_".join(i.split("_")[:-1]) in dataset_config.filter_categories]
                for split, ids in datasplits.items()
            }

        # Reuse the .pt dataset's "train[0:100]" subset notation.
        self.split, subset_indices = DrivAerNetDataset._parse_split_subset(dataset_config.split)
        if self.split not in datasplits:
            raise ValueError(f"Unknown split '{self.split}'. Available splits: {list(datasplits)}")
        all_design_ids = datasplits[self.split]
        self.design_ids = [all_design_ids[i] for i in subset_indices] if subset_indices else all_design_ids

        missing = [design_id for design_id in self.design_ids if design_id not in self.manifest.samples]
        if missing:
            raise KeyError(
                f"{len(missing)} '{self.split}' samples missing from the Zarr store "
                f"(e.g. {missing[0]}). Re-run conversion to include them."
            )
        logger.info(
            "Initialized ZarrDrivAerNetDataset with %d samples for split '%s'", len(self.design_ids), self.split
        )

    def _read_lines(self, name: str) -> list[str]:
        with fsspec.open(f"{str(self.store_root).rstrip('/')}/{name}", "r") as f:
            return [line.strip() for line in f if line.strip()]

    def _load_datasplits(self) -> dict[str, list[str]]:
        """Load split design ids from the store root, excluding blacklisted designs."""
        blacklist = {
            line.split("/")[-1].split(".vtk")[0]
            for name in ("blacklist.txt", "blacklist2.txt")
            for line in self._read_lines(name)
        }
        return {
            split: [
                design_id for design_id in self._read_lines(f"{split}_design_ids.txt") if design_id not in blacklist
            ]
            for split in ("train", "test", "val")
        }

    def __len__(self) -> int:
        return len(self.design_ids)

    def _sample_id(self, idx: int) -> str:
        return str(self.design_ids[idx % len(self.design_ids)])

    def sample_info(self, idx: int) -> dict[str, str | int | None]:
        """Get information about a sample such as its store path and design id."""
        idx = idx % len(self.design_ids)
        design_id = self.design_ids[idx]
        return {
            "sample_uri": stores.join(self.store_root, self.manifest.samples[design_id].relpath),
            "run_name": design_id,
            "design_id": design_id,
            "split": self.split,
        }
