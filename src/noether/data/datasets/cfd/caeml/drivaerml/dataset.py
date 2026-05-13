#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
from pathlib import Path

from noether.data.base.dataset import DatasetSplitIDs, StandardDatasetConfig
from noether.data.datasets.cfd.caeml.dataset import CAEMLDataset
from noether.data.datasets.cfd.caeml.drivaerml.split import DrivAerMLDefaultSplitIDs

logger = logging.getLogger(__name__)


class DrivAerMLDataset(CAEMLDataset):
    """
    Dataset implementation for DrivaerML CFD simulations.

    Args:
        dataset_config: Configuration for the dataset.
    """

    STATS_FILE: str = str(Path(__file__).parent / "stats.yaml")

    def __init__(
        self,
        dataset_config: StandardDatasetConfig,
    ):
        """
        Initialize the DrivaerML dataset.

        Args:
            dataset_config: Configuration for the dataset.

        """
        assert self.get_dataset_splits.DATASET_NAME is not None
        super().__init__(dataset_config=dataset_config, dataset_name=self.get_dataset_splits.DATASET_NAME)

    @property
    def get_dataset_splits(self) -> DatasetSplitIDs:
        return DrivAerMLDefaultSplitIDs()
