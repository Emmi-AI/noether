#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import torch

from noether.data.base.dataset import StandardDatasetConfig
from noether.data.datasets.cfd.caeml.drivaerml.dataset import DrivAerMLDataset
from noether.data.datasets.cfd.shapenet_car.dataset import ShapeNetCarDataset
from noether.data.datasets.cfd.shapenet_car.filemap import SHAPENET_CAR_FILEMAP
from noether.data.datasets.cfd.simshift_heatsink.config import SimshiftHeatsinkConfig
from noether.data.datasets.cfd.simshift_heatsink.dataset import SimshiftHeatsinkDataset


def _seeded_randn(*shape: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen)


def _seeded_unit_vectors(*shape: int, seed: int) -> torch.Tensor:
    return torch.nn.functional.normalize(_seeded_randn(*shape, seed=seed), dim=-1)


def _seeded_positions(n: int, dim: int, *, low: float, high: float, seed: int) -> torch.Tensor:
    """Uniform points in ``[low, high]^dim``.

    The production ``PositionNormalizer`` rejects values outside the dataset's ``raw_pos_min`` / ``raw_pos_max``
    bounds, so synthetic positions must stay inside that bbox or the test fails before any forward pass.
    """
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(n, dim, generator=gen) * (high - low) + low


class StubShapeNetCarDataset(ShapeNetCarDataset):
    """In-memory ShapeNetCarDataset that bypasses all disk I/O.

    Returns synthetic tensors sized to satisfy the pipeline's downsampling requirements (``num_surface_points=3586``,
    ``num_volume_points=4096`` per the ``ShapeNetCarPreset``).
    """

    NUM_SAMPLES: int = 4
    NUM_SURFACE_POINTS: int = 4096
    NUM_VOLUME_POINTS: int = 4096
    POS_MIN: float = -4.5
    POS_MAX: float = 6.0

    def __init__(self, dataset_config: StandardDatasetConfig) -> None:
        super().__init__(dataset_config=dataset_config)

    def _resolve_source_root_path(self) -> None:
        return

    def _load_design_ids(self) -> None:
        self.design_ids = [f"stub/{i:04d}" for i in range(self.NUM_SAMPLES)]

    def _load(self, idx: int, filename: str) -> torch.Tensor:
        n_surf = self.NUM_SURFACE_POINTS
        n_vol = self.NUM_VOLUME_POINTS
        seed = (hash((type(self).__name__, idx, filename)) & 0x7FFFFFFF) or 1

        if filename == SHAPENET_CAR_FILEMAP.surface_position:
            return _seeded_positions(n_surf, 3, low=self.POS_MIN, high=self.POS_MAX, seed=seed)
        if filename == SHAPENET_CAR_FILEMAP.surface_pressure:
            return _seeded_randn(n_surf, seed=seed)
        if filename == SHAPENET_CAR_FILEMAP.surface_normals:
            return _seeded_unit_vectors(n_surf, 3, seed=seed)
        if filename == SHAPENET_CAR_FILEMAP.volume_position:
            return _seeded_positions(n_vol, 3, low=self.POS_MIN, high=self.POS_MAX, seed=seed)
        if filename == SHAPENET_CAR_FILEMAP.volume_velocity:
            return _seeded_randn(n_vol, 3, seed=seed)
        if filename == SHAPENET_CAR_FILEMAP.volume_distance_to_surface:
            return _seeded_randn(n_vol, seed=seed)
        if filename == SHAPENET_CAR_FILEMAP.volume_normals:
            return _seeded_unit_vectors(n_vol, 3, seed=seed)
        raise KeyError(f"StubShapeNetCarDataset has no synthetic tensor for filename={filename!r}")


class StubDrivAerMLDataset(DrivAerMLDataset):
    """In-memory DrivAerMLDataset that bypasses all disk I/O.

    Sized for the ``DrivAerMLPreset`` pipeline (``num_surface_points=16384``, ``num_volume_points=16384``).
    """

    NUM_SAMPLES: int = 4
    NUM_SURFACE_POINTS: int = 16384
    NUM_VOLUME_POINTS: int = 16384
    POS_MIN: float = -40.0
    POS_MAX: float = 80.0

    def __init__(self, dataset_config: StandardDatasetConfig) -> None:
        super().__init__(dataset_config=dataset_config)

    def _load_design_ids(self) -> None:
        self.design_ids = [str(i) for i in range(self.NUM_SAMPLES)]

    def _load(self, idx: int, filename: str) -> torch.Tensor:
        n_surf = self.NUM_SURFACE_POINTS
        n_vol = self.NUM_VOLUME_POINTS
        seed = (hash((type(self).__name__, idx, filename)) & 0x7FFFFFFF) or 1
        fm = self.filemap

        if filename == fm.surface_position:
            return _seeded_positions(n_surf, 3, low=self.POS_MIN, high=self.POS_MAX, seed=seed)
        if filename == fm.surface_pressure:
            return _seeded_randn(n_surf, seed=seed)
        if filename == fm.surface_friction:
            return _seeded_randn(n_surf, 3, seed=seed)
        if filename == fm.surface_normals:
            return _seeded_unit_vectors(n_surf, 3, seed=seed)
        if filename == fm.surface_area:
            return _seeded_randn(n_surf, seed=seed).abs()
        if filename == fm.volume_position:
            return _seeded_positions(n_vol, 3, low=self.POS_MIN, high=self.POS_MAX, seed=seed)
        if filename == fm.volume_pressure:
            return _seeded_randn(n_vol, seed=seed)
        if filename == fm.volume_velocity:
            return _seeded_randn(n_vol, 3, seed=seed)
        if filename == fm.volume_vorticity:
            return _seeded_randn(n_vol, 3, seed=seed)
        raise KeyError(f"StubDrivAerMLDataset has no synthetic tensor for filename={filename!r}")


class StubSimshiftHeatsinkDataset(SimshiftHeatsinkDataset):
    """In-memory SimshiftHeatsinkDataset bypassing HDF5 / HuggingFace download.

    Mirrors the production dataset's ``pre_getitem`` contract: returns a dict
    with ``position``/``velocity``/``temperature``/``pressure`` per sample. The
    ``simulation_parameters`` getter is also overridden since it reads from a
    pandas DataFrame the real ``__init__`` builds.
    """

    NUM_SAMPLES: int = 4
    NUM_VOLUME_POINTS: int = 4096
    NUM_SIM_PARAMS: int = 5

    def __init__(self, dataset_config: SimshiftHeatsinkConfig) -> None:
        # Fully bypass the real __init__ (HDF5 / HuggingFace download / metadata
        # CSV). Reproduce the minimum invariants: dataset_config wiring through
        # the grandparent Dataset.__init__, plus the attributes that getitem_*
        # methods read.
        from noether.data.base.dataset import Dataset as _BaseDataset

        _BaseDataset.__init__(self, dataset_config=dataset_config)
        self._zip_path = None
        self.source_root = None  # type: ignore[assignment]
        self.difficulty = getattr(dataset_config, "difficulty", "easy")
        self.domain = getattr(dataset_config, "domain", "source")
        self.split = dataset_config.split
        self._sample_ids = list(range(self.NUM_SAMPLES))
        self._cond_columns = [f"param_{i}" for i in range(self.NUM_SIM_PARAMS)]

    # Per-axis bbox derived from the production stats.json (volume_position_min /
    # volume_position_max). Positions outside this box trip the FieldNormalizer's
    # bounds check.
    POS_LOW: tuple[float, float, float] = (-0.069, -0.069, 0.001)
    POS_HIGH: tuple[float, float, float] = (0.069, 0.069, 0.499)

    def _read_h5(self, sample_id: int) -> dict[str, torch.Tensor]:
        n = self.NUM_VOLUME_POINTS
        seed = (hash((type(self).__name__, sample_id)) & 0x7FFFFFFF) or 1
        gen = torch.Generator().manual_seed(seed)
        low = torch.tensor(self.POS_LOW)
        high = torch.tensor(self.POS_HIGH)
        position = torch.rand(n, 3, generator=gen) * (high - low) + low
        return {
            "position": position,
            "velocity": _seeded_randn(n, 3, seed=seed + 1),
            "temperature": _seeded_randn(n, 1, seed=seed + 2),
            "pressure": _seeded_randn(n, 1, seed=seed + 3),
        }

    def getitem_simulation_parameters(self, idx: int) -> torch.Tensor:  # type: ignore[override]
        seed = (hash((type(self).__name__, idx, "sim_params")) & 0x7FFFFFFF) or 1
        return _seeded_randn(1, self.NUM_SIM_PARAMS, seed=seed)
