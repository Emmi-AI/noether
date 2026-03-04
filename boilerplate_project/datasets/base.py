#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import math
from dataclasses import dataclass

import torch

from boilerplate_project.schemas.datasets.base_dataset_config import BaseDatasetConfig
from noether.data import Dataset


@dataclass
class BaseDatasetStats:
    """Not used at the moment but can be used to store normalization stats for the dataset."""

    mean: float = (0.0,)
    variance: float = (1.0,)


class BaseDataset(Dataset):
    """A base dataset implementation for the `noether` framework that generates synthetic data for multi-class classification."""

    def __init__(
        self,
        dataset_config: BaseDatasetConfig,
        **kwargs,
    ):
        super().__init__(dataset_config=dataset_config, **kwargs)

        self.num_samples = dataset_config.num_samples
        self.num_features = 2
        self.num_classes = dataset_config.num_classes
        self.noise = dataset_config.noise
        self.radius = dataset_config.radius

        self._generate_data()

    def _generate_data(self) -> torch.Tensor:
        """
        Generates distinct clusters of data for multi-class classification using PyTorch.

        The cluster centers are arranged evenly on a circle.
        """

        all_X = []
        all_y = []
        samples_per_class = self.num_samples // self.num_classes

        for i in range(self.num_classes):
            # Calculate the angle for the current class center on a circle
            angle = (i / self.num_classes) * 2 * math.pi

            # Calculate the center coordinates
            center_x = self.radius * math.cos(angle)
            center_y = self.radius * math.sin(angle)
            center = torch.tensor([center_x, center_y])

            # Generate points for the current class
            # torch.randn creates random data, which we scale by noise and move to the center
            class_X = torch.randn(samples_per_class, self.num_features) * self.noise + center

            # Create labels for the current class
            # torch.full creates a tensor filled with the class index 'i'
            class_y = torch.full((samples_per_class,), fill_value=i, dtype=torch.long)

            all_X.append(class_X)
            all_y.append(class_y)

        # Concatenate all class data into single tensors
        X = torch.cat(all_X, dim=0)
        y = torch.cat(all_y, dim=0)

        self.x = X
        self.y = y

    @staticmethod
    def get_normalization_stats() -> BaseDatasetStats:
        """Returns the normalization stats for the dataset. This is used in the pipeline to normalize the data."""
        return BaseDatasetStats()

    def __len__(self) -> int:
        """returns the number of samples in the dataset."""
        return self.num_samples

    def getitem_x(self, index: int) -> torch.Tensor:
        """Returns the input data for the given index."""
        return self.x[index]

    def getitem_y(self, index: int) -> torch.Tensor:
        """Returns the output data for the given index."""
        return self.y[index]
