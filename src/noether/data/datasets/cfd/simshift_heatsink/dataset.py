#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import json
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from noether.core.utils.common import validate_path
from noether.data import Dataset, with_normalizers
from noether.data.datasets.cfd.simshift_heatsink.config import SimshiftHeatsinkConfig

logger = logging.getLogger(__name__)

# Columns dropped from metadata to form the conditioning vector.
# These are all constant across the dataset and not useful for conditioning.
_DROPPED_METADATA_COLUMNS = frozenset(
    {
        "sample_id",
        "envTemp",
        "flowVelocity",
        "height1",
        "length",
        "pressure",
        "turbulentKE",
        "turbulentOmega",
        "width",
    }
)

# HDF5 field names for the heatsink dataset.
_FIELD_VELOCITY = "U"
_FIELD_TEMPERATURE = "T"
_FIELD_PRESSURE = "p_rgh"


class SimshiftHeatsinkDataset(Dataset):
    """Dataset for the SIMSHIFT Heatsink CFD benchmark.

    The SIMSHIFT Heatsink dataset contains conjugate heat transfer simulations of heatsink
    geometries with varying fin configurations. Data is stored in HDF5 format with mesh
    coordinates and element-level physical fields (velocity, temperature, pressure).

    The dataset supports source/target domain splits at different difficulty levels
    for unsupervised domain adaptation experiments.

    Reference: https://arxiv.org/abs/2506.12007
    """

    def __init__(self, dataset_config: SimshiftHeatsinkConfig) -> None:
        super().__init__(dataset_config=dataset_config)

        self.source_root = validate_path(dataset_config.root)
        self.difficulty = dataset_config.difficulty
        self.domain = dataset_config.domain
        self.split = dataset_config.split

        splits_path = (
            Path(dataset_config.splits_path) if dataset_config.splits_path else self.source_root / "splits.json"
        )
        metadata_path = (
            Path(dataset_config.metadata_path) if dataset_config.metadata_path else self.source_root / "metadata.csv"
        )

        self._metadata_df = pd.read_csv(metadata_path)
        self._cond_columns = [c for c in self._metadata_df.columns if c not in _DROPPED_METADATA_COLUMNS]

        self._sample_ids = self._load_split_ids(splits_path)

        logger.info(
            "Initialized SimshiftHeatsinkDataset (%s/%s/%s) with %d samples",
            self.difficulty,
            self.domain,
            self.split,
            len(self._sample_ids),
        )

    def _load_split_ids(self, splits_path: Path) -> list[int]:
        """Load sample IDs for the configured difficulty/domain/split."""
        with open(splits_path) as f:
            splits_metadata = json.load(f)

        difficulties = [self.difficulty] if self.difficulty is not None else list(splits_metadata.keys())

        all_ids: list[int] = []
        for diff in difficulties:
            if self.domain is not None:
                domain_keys = ["src" if self.domain == "source" else "tgt"]
            else:
                domain_keys = ["src", "tgt"]

            for domain_key in domain_keys:
                split_key = self.split
                # Target domain has no validation split; fall back to test.
                if domain_key == "tgt" and self.split == "val":
                    split_key = "test"
                all_ids.extend(splits_metadata[diff][domain_key][split_key])

        return all_ids

    def __len__(self) -> int:
        return len(self._sample_ids)

    def pre_getitem(self, idx: int) -> dict[str, torch.Tensor]:
        """Load all fields for sample *idx* from its HDF5 file.

        The returned dict is forwarded as kwargs to every ``getitem_*`` method.
        """
        sample_id = self._sample_ids[idx]
        h5_path = self.source_root / f"{sample_id}.h5"

        with h5py.File(h5_path, "r") as h5f:
            # Element centre coordinates
            coords = torch.from_numpy(h5f["mesh/element_coords"][:, :]).float()

            # Velocity (num_elements, 3)
            vel = torch.from_numpy(h5f[f"element_fields/{_FIELD_VELOCITY}"][:]).float()
            if vel.ndim == 1:
                vel = vel.unsqueeze(-1)

            # Temperature (num_elements, 1)
            temp = torch.from_numpy(h5f[f"element_fields/{_FIELD_TEMPERATURE}"][:]).float()
            if temp.ndim == 1:
                temp = temp.unsqueeze(-1)

            # Pressure (num_elements, 1)
            pres = torch.from_numpy(h5f[f"element_fields/{_FIELD_PRESSURE}"][:]).float()
            if pres.ndim == 1:
                pres = pres.unsqueeze(-1)

        return {"position": coords, "velocity": vel, "temperature": temp, "pressure": pres}

    @with_normalizers
    def getitem_volume_position(self, idx: int, *, position: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        """Element centre coordinates of the volume mesh (num_elements, 3)."""
        return position

    @with_normalizers
    def getitem_volume_velocity(self, idx: int, *, velocity: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        """Velocity field at element centres (num_elements, 3)."""
        return velocity

    @with_normalizers
    def getitem_volume_temperature(self, idx: int, *, temperature: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        """Temperature field at element centres (num_elements, 1)."""
        return temperature

    @with_normalizers
    def getitem_volume_pressure(self, idx: int, *, pressure: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        """Pressure (p_rgh) field at element centres (num_elements, 1)."""
        return pressure

    @with_normalizers
    def getitem_simulation_parameters(self, idx: int) -> torch.Tensor:
        """Geometry design parameters conditioning vector (num_params,)."""
        sample_id = self._sample_ids[idx]
        row = self._metadata_df[self._metadata_df["sample_id"] == sample_id]
        cond_np = row[self._cond_columns].iloc[0].to_numpy(dtype=np.float32)
        return torch.from_numpy(cond_np).unsqueeze(0)

    def sample_info(self, idx: int) -> dict[str, str | int | None]:
        """Get information about a sample such as its path, sample ID, etc."""
        sample_id = self._sample_ids[idx]
        return {
            "sample_uri": str(self.source_root / f"{sample_id}.h5"),
            "sample_id": sample_id,
            "difficulty": self.difficulty,
            "domain": self.domain,
            "split": self.split,
        }
