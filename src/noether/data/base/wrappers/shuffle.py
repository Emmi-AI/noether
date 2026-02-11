#  Copyright © 2025 Emmi AI GmbH. All rights reserved.


import numpy as np

from noether.core.schemas.dataset import ShuffleWrapperConfig
from noether.data.base import Dataset, Subset


class ShuffleWrapper(Subset):
    """Shuffles the dataset, optionally with seed."""

    def __init__(self, dataset: Dataset, config: ShuffleWrapperConfig):
        """
        Args:
            dataset: The dataset to shuffle.
            config: Configuration for the ShuffleWrapper. See :class:`~noether.core.schemas.dataset.ShuffleWrapperConfig`
                for available options.
        Raises:
            ValueError: If the dataset is not an instance of noether.data.Dataset, or if the seed is not an integer or None.
        """
        if not isinstance(dataset, Dataset):
            raise ValueError("The dataset must be an instance of noether.data.Dataset.")
        self.seed = config.seed

        if self.seed is not None:
            rng = np.random.default_rng(seed=self.seed)
        else:
            rng = np.random  # type: ignore
        indices = np.arange(len(dataset), dtype=int)
        rng.shuffle(indices)
        super().__init__(dataset=dataset, indices=indices)  # type: ignore

    def __str__(self) -> str:
        dataset_str = (
            str(self.dataset.__class__.__name__) if self.dataset.__str__ is object.__str__ else str(self.dataset)
        )
        return f"{dataset_str} (shuffled with seed={self.seed})"
