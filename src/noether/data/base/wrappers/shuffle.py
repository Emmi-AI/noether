#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field

from noether.data.base.subset import Subset
from noether.data.base.wrapper import DatasetWrapper, DatasetWrapperConfig

if TYPE_CHECKING:
    from noether.data.base.dataset import Dataset


class ShuffleWrapperConfig(DatasetWrapperConfig):
    seed: int | None = Field(None, ge=0)
    """Random seed for shuffling. If None, a random seed is used."""


class ShuffleWrapper(Subset):
    """Shuffles the dataset, optionally with seed.

    Example:

        .. code-block:: yaml

            dataset_wrappers:
                kind: noether.data.base.wrappers.ShuffleWrapper
                seed: 42
    """

    def __init__(self, config: ShuffleWrapperConfig, dataset: Dataset | DatasetWrapper):
        """
        Args:
            config: Configuration for the ShuffleWrapper. See :class:`~noether.core.schemas.dataset.ShuffleWrapperConfig`
                for available options.
            dataset: The dataset to shuffle. Can be a base dataset or an already wrapped dataset.

        Raises:
            ValueError: If the dataset is not an instance of noether.data.Dataset or DatasetWrapper, or if the seed is not an integer or None.
        """
        from noether.data.base.dataset import Dataset

        if not isinstance(dataset, (Dataset, DatasetWrapper)):
            raise ValueError("The dataset must be an instance of noether.data.Dataset or noether.data.DatasetWrapper.")
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
