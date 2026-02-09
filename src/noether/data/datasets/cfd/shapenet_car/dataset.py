#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
from pathlib import Path

import torch

from noether.core.schemas.dataset import DatasetBaseConfig, DatasetSplitIDs
from noether.core.utils.common import validate_path
from noether.data.datasets.cfd.dataset import AeroDataset
from noether.data.datasets.cfd.shapenet_car.filemap import SHAPENET_CAR_FILEMAP
from noether.data.datasets.cfd.shapenet_car.split import ShapeNetCarDefaultSplitIDs

logger = logging.getLogger(__name__)

NUM_PARAM_FOLDERS = 9
PREPROCESSED_FOLDER_NAME = "preprocessed"
TEST_PARAM_INDEX = 0
SUPPORTED_SPLITS = {"train", "test"}


class ShapeNetCarDataset(AeroDataset):
    """
    Dataset implementation for ShapeNet Car CFD simulations.

    This dataset provides access to:
    - Surface properties: positions, pressure, normals
    - Volume properties: positions, velocity, normals, signed distance field (SDF)

    The dataset is split by parameter configurations:
    - Test: param0 (100 samples)
    - Validation: no validation split defined
    - Train: param1-8 (789 samples)

    Download link to the raw dataset: http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip

    Expected directory structure:
        root/
            preprocessed/
                param0/
                    <simulation_id>/
                        surface_points.pt
                        surface_pressure.pt
                        surface_normals.pt
                        volume_velocity.pt
                        volume_points.pt
                        volume_sdf.pt
                        volume_normals.pt
                param1/
                    ...
                ...
                param8/
    """

    def __init__(
        self,
        dataset_config: DatasetBaseConfig,
    ):
        """
        Initialize the ShapeNet Car dataset.

        Args:
            dataset_config: Configuration for the dataset.

        Raises:
            TypeError: If dataset_config is not ShapeNetDatasetConfig
            ValueError: If configuration is invalid or split is unknown
            FileNotFoundError: If data directory does not exist
        """
        super().__init__(dataset_config=dataset_config, filemap=SHAPENET_CAR_FILEMAP)

        self.split = dataset_config.split
        if self.split not in SUPPORTED_SPLITS:
            raise ValueError(f"Unsupported split '{self.split}'. Supported splits are: {SUPPORTED_SPLITS}")
        self.source_root: Path = validate_path(dataset_config.root)  # type: ignore[arg-type]
        self._resolve_source_root_path()
        self._load_design_ids()
        logger.info(f"Initialized ShapeNetCarDataset with {len(self.design_ids)} samples for split '{self.split}'")

    def _load_design_ids(self) -> None:
        """
        Load URIs for dataset samples based on the split.

        Following the Transolver paper convention:
        - param0 is used for test/validation set
        - param1-8 are used for training set

        Note:
            Alternative approach would be to define splits via explicit file lists.
        """
        self.design_ids = list(self.get_dataset_splits.model_dump()[self.split])

    def _resolve_source_root_path(self) -> None:
        """
        Resolve the root path to the preprocessed folder and validate it exists.

        If the provided path doesn't end with 'preprocessed', appends it.

        Raises:
            FileNotFoundError: If the resolved path does not exist
            ValueError: If the resolved path is not a 'preprocessed' folder
        """
        if self.source_root.name != PREPROCESSED_FOLDER_NAME:
            self.source_root = self.source_root / PREPROCESSED_FOLDER_NAME

        if not self.source_root.exists():
            raise FileNotFoundError(f"Preprocessed data folder does not exist: {self.source_root.as_posix()}")

        if self.source_root.name != PREPROCESSED_FOLDER_NAME:
            raise ValueError(f"Expected '{PREPROCESSED_FOLDER_NAME}' folder, but got '{self.source_root.name}'")

    @property
    def get_dataset_splits(self) -> DatasetSplitIDs:
        return ShapeNetCarDefaultSplitIDs()

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            Number of simulation samples available
        """
        return len(self.design_ids)

    def _load(self, idx: int, filename: str) -> torch.Tensor:
        """
        Load a tensor from a specific file in a sample directory.

        Args:
            idx: Index of the sample to load (automatically wrapped with modulo)
            filename: Name of the file to load from the sample directory

        Returns:
            Loaded tensor from the specified file

        Raises:
            FileNotFoundError: If the requested file does not exist
            RuntimeError: If loading fails for any reason
        """

        try:
            sample_uri = self.source_root / self.design_ids[idx] / filename
            return torch.load(sample_uri, weights_only=True)  # type: ignore[no-any-return]
        except Exception as e:
            raise RuntimeError(f"Failed to load {sample_uri}: {e}") from e

    def sample_info(self, idx: int) -> dict[str, str | int | None]:
        """Get information about a sample such as its local path, run name, etc."""
        idx = idx % len(self.design_ids)
        design_id = self.design_ids[idx]
        sample_uri = self.source_root / design_id
        return {
            "sample_uri": sample_uri,
            "run_name": design_id,
            "split": self.split,
        }
