#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Zarr-backed ShapeNet-Car dataset with chunk-based subsampling."""

from __future__ import annotations

import logging

from noether.data.datasets.cfd.shapenet_car.dataset import SUPPORTED_SPLITS, ShapeNetCarDataset
from noether.data.datasets.cfd.shapenet_car.filemap import SHAPENET_CAR_FILEMAP
from noether.data.datasets.cfd.shapenet_car.split import ShapeNetCarDefaultSplitIDs
from noether.data.datasets.cfd.zarr_aero_dataset import ZarrAeroDataset, ZarrAeroDatasetConfig
from noether.data.zarr_store import stores

logger = logging.getLogger(__name__)


class ZarrShapeNetCarDatasetConfig(ZarrAeroDatasetConfig):
    """Config for the Zarr-backed ShapeNet-Car dataset (``root`` = converted Zarr store)."""

    kind: str | None = "noether.data.datasets.cfd.ZarrShapeNetCarDataset"


class ZarrShapeNetCarDataset(ZarrAeroDataset):
    """ShapeNet-Car dataset that reads from a converted Zarr store.

    Config-driven analogue of :class:`~noether.data.datasets.cfd.shapenet_car.dataset.ShapeNetCarDataset`
    where ``dataset_config.root`` points at the converted Zarr store. When the config's
    ``num_surface_points`` / ``num_volume_points`` are given, the dataset chunk-subsamples
    at read time and returns exactly that many points per domain, so the pipeline's
    ``PointSamplingSampleProcessor`` becomes a no-op.
    """

    STATS_FILE: str = ShapeNetCarDataset.STATS_FILE

    def __init__(self, dataset_config: ZarrShapeNetCarDatasetConfig) -> None:
        """

        Args:
            dataset_config: ShapeNet-Car Zarr config; ``root`` is the Zarr store root and the
                ``num_*`` / ``sampling_seed`` / ``read_concurrency`` fields drive subsampling.
        """
        super().__init__(
            dataset_config=dataset_config,
            filemap=SHAPENET_CAR_FILEMAP,
            num_points={
                "surface": dataset_config.num_surface_points,
                "volume": dataset_config.num_volume_points,
            },
            sampling_seed=dataset_config.sampling_seed,
            read_concurrency=dataset_config.read_concurrency,
            num_geometry_points=dataset_config.num_geometry_points,
        )
        self.split = dataset_config.split
        if self.split not in SUPPORTED_SPLITS:
            raise ValueError(f"Unsupported split '{self.split}'. Supported splits are: {SUPPORTED_SPLITS}")
        self.design_ids = list(getattr(ShapeNetCarDefaultSplitIDs(), self.split))
        missing = [d for d in self.design_ids if d not in self.manifest.samples]
        if missing:
            raise KeyError(
                f"{len(missing)} '{self.split}' samples missing from the Zarr store "
                f"(e.g. {missing[0]}). Re-run conversion to include them."
            )
        logger.info(
            "Initialized ZarrShapeNetCarDataset with %d samples for split '%s'", len(self.design_ids), self.split
        )

    def __len__(self) -> int:
        return len(self.design_ids)

    def _sample_id(self, idx: int) -> str:
        return str(self.design_ids[idx % len(self.design_ids)])

    def sample_info(self, idx: int) -> dict[str, str | int | None]:
        """Get information about a sample such as its store path and run name."""
        idx = idx % len(self.design_ids)
        design_id = self.design_ids[idx]
        return {
            "sample_uri": stores.join(self.store_root, self.manifest.samples[design_id].relpath),
            "run_name": design_id,
            "split": self.split,
        }
