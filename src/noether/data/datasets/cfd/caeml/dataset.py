#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging

import torch

from noether.core.schemas.dataset import DatasetBaseConfig, DatasetSplitIDs
from noether.core.utils.common import validate_path
from noether.data.datasets.cfd.caeml.filemap import CAEML_FILEMAP
from noether.data.datasets.cfd.dataset import AeroDataset

logger = logging.getLogger(__name__)


class CAEMLDataset(AeroDataset):
    """
    Dataset implementation for CAEML datasets AhmedML and DrivAerML.

    Args:
        dataset_config: Configuration for the dataset.
        dataset_name: Name of the dataset.
    """

    def __init__(
        self,
        dataset_config: DatasetBaseConfig,
        dataset_name: str,
        filemap=CAEML_FILEMAP,
    ):
        """

        Args:
            dataset_config: Configuration for the dataset.
            dataset_name: Name of the dataset.

        Raises:
            TypeError: If dataset_config is not ShapeNetDatasetConfig
            ValueError: If configuration is invalid or split is unknown
            FileNotFoundError: If data directory does not exist
        """
        super().__init__(dataset_config=dataset_config, filemap=filemap)

        self.split = dataset_config.split
        if self.split not in self.supported_splits:
            raise ValueError(f"Unsupported split '{self.split}'. Supported splits are: {self.supported_splits}")
        self.source_root = validate_path(dataset_config.root)  # type: ignore[arg-type]
        self._load_design_ids()
        logger.info(f"Initialized {dataset_name} with {len(self.design_ids)} samples for split '{self.split}'")

    def _load_design_ids(self) -> None:
        """
        Load URIs for dataset samples based on the split.
        """
        self.design_ids = list(self.get_dataset_splits.model_dump()[self.split])

    @property
    def supported_splits(self) -> set[str]:
        return {"train", "test", "val"}

    @property
    def get_dataset_splits(self) -> DatasetSplitIDs:
        raise NotImplementedError("Subclasses must implement get_dataset_splits")

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

        sample_uri = self.source_root / f"run_{self.design_ids[idx]}" / filename
        try:
            return torch.load(sample_uri, weights_only=True)  # type: ignore[no-any-return]
        except Exception as e:
            raise RuntimeError(f"Failed to load {sample_uri}, {filename}: {e}") from e

    def sample_info(self, idx: int) -> dict[str, str | int | None]:
        """Get information about a sample such as its local path, run name, etc."""
        idx = idx % len(self.design_ids)
        design_id = self.design_ids[idx]
        run_name = f"run_{design_id}"
        sample_uri = self.source_root / run_name
        return {
            "sample_uri": sample_uri,
            "run_name": run_name,
            "run_number": design_id,
            "design_id": design_id,
            "split": self.split,
        }
