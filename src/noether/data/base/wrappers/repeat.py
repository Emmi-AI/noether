#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import numpy as np

from noether.core.schemas.dataset import RepeatWrapperConfig
from noether.data.base import Dataset, Subset


class RepeatWrapper(Subset):
    """Repeats the wrapped dataset `repetitions` times.

    Example:

    .. code-block:: python

        from noether.data import Dataset as ListDataset

        dataset = ListDataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        len(dataset)
        10
        repeat_dataset = RepeatWrapper(dataset, repetitions=3)
        len(repeat_dataset)
        30
    """

    def __init__(self, config: RepeatWrapperConfig, dataset: Dataset) -> None:
        """

        Args:
            config: Configuration for the RepeatWrapper. See :class:`~noether.core.schemas.dataset.RepeatWrapperConfig`
                for available options.
            dataset: The dataset to repeat.
        Raises:
            ValueError: If repetitions is less than 2 or if the dataset is empty.
                You don't need to use this wrapper with repetitions < 2.
        """

        if len(dataset) == 0:
            raise ValueError("The dataset is empty.")

        self.repetitions = config.repetitions
        # repeat indices <repetitions> times in round-robin fashion (indices are like [0, 1, 2, 0, 1, 2])
        indices = np.tile(np.arange(len(dataset), dtype=int), self.repetitions)
        super().__init__(dataset=dataset, indices=indices)  # type: ignore

    def __str__(self) -> str:
        dataset_str = (
            str(self.dataset.__class__.__name__) if self.dataset.__str__ is object.__str__ else str(self.dataset)
        )
        return f"{dataset_str} (repeated {self.repetitions} times)"
