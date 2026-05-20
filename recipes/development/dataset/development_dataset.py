#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import torch

from noether.data import Dataset, with_normalizers
from noether.data.base.dataset import StandardDatasetConfig


class DevelopmentDatasetConfig(StandardDatasetConfig):
    root: str | None = None
    x_dim: int
    y_dim: int
    z_dim: int
    sample_size: int
    num_samples: int


class DevelopmentDataset(Dataset):
    def __init__(self, dataset_config: DevelopmentDatasetConfig):
        super().__init__(dataset_config=dataset_config)
        self.dataset_config = dataset_config

    def __len__(self):
        return self.dataset_config.num_samples

    @with_normalizers
    def getitem_x(self, idx: int):
        return torch.randn((self.dataset_config.sample_size, self.dataset_config.x_dim))

    @with_normalizers
    def getitem_y(self, idx: int):
        return torch.randn((self.dataset_config.sample_size, self.dataset_config.y_dim))

    @with_normalizers
    def getitem_z(self, idx: int):
        return torch.randn((self.dataset_config.sample_size, self.dataset_config.z_dim))
