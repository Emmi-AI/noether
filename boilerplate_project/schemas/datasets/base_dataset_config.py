#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from noether.core.schemas.dataset import DatasetBaseConfig


class BaseDatasetConfig(DatasetBaseConfig):
    num_samples: int
    """Total number of samples to generate."""
    num_classes: int = 10
    """The number of distinct classes (clusters) to generate."""
    noise: float = 0.1
    """The standard deviation of the Gaussian noise added to the data."""
    radius: float = 1.0
    """The radius of the circle on which the cluster centers are placed."""
