#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import torch

from noether.core.schemas.dataset import DatasetBaseConfig
from noether.data import Dataset

from .pipeline import PipelineConfig


class DatasetConfig(DatasetBaseConfig):
    # custom dataset config options can be added here
    pipeline: PipelineConfig


class Dataset(Dataset):
    def __init__(self, dataset_config: DatasetConfig, **kwargs):
        super().__init__(dataset_config=dataset_config, **kwargs)

    def __len__(self) -> int:
        # return the length of your dataset here
        raise NotImplementedError("Implement the __len__ method to return the length of your dataset.")

    def getitem_(self, index: int) -> dict[str, torch.Tensor]:
        # implement the logic to get a single item from your dataset here
        raise NotImplementedError("Implement the getitem_ method to return a single item from your dataset.")
