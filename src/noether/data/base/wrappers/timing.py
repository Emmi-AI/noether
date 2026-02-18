#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

from torch.utils.data import Dataset

from noether.core.utils.common.stopwatch import Stopwatch

META_GETITEM_TIME = "__meta_time_getitem"


class TimingWrapper(Dataset):
    """Wrapper that times __getitem__ calls and returns both the item and the time taken."""

    def __init__(self, dataset):
        """
        Args:
            dataset: The dataset to wrap
        """
        self.dataset = dataset

    def __getitem__(self, index):
        """
        Get item and measure time taken.

        Returns:
            tuple: (item, time_taken) where time_taken is in seconds
        """
        with Stopwatch() as sw:
            item = self.dataset[index]
        if isinstance(item, dict):
            item[META_GETITEM_TIME] = sw.elapsed_seconds
            return item
        return item

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, item: str) -> Any:
        """Wraps __getitems__ methods to disable batched data retrieval. Other properties are also directly
        passed-through to allow property access independent of whether the dataset is wrapped into a PropertySubsetWrapper or
        not. For example, if a classification dataset contains the property num_classes, wrapping that dataset would
        require dataset.dataset.num_classes instead of dataset.num_classes to access the field. This method makes sure
        that the wrapper is fully transparent and the num_classes field can be accessed with dataset.num_classes.

        Args:
            item (str): name of the attribute to access (e.g., "getitem_x" to access dataset.getitem_x)

        Returns:
            Any: the result of the getitem_ (using the subset indices) or the attribute of the base dataset.
        """
        if item == "dataset":
            return getattr(super(), item)
        return getattr(self.dataset, item)

    def __str__(self) -> str:
        return str(self.dataset)
