#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

from noether.core.utils.common.stopwatch import Stopwatch
from noether.data.base import Dataset, DatasetWrapper

META_GETITEM_TIME = "__meta_time_getitem"


class TimingWrapper(DatasetWrapper):
    """Wrapper that times __getitem__ calls and returns both the item and the time taken."""

    def __init__(self, dataset: Dataset | DatasetWrapper):
        """
        Args:
            dataset: The dataset to wrap
        """
        super().__init__(dataset=dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Get item and measure time taken.

        Returns:
            Dictionary sample extended with ``META_GETITEM_TIME``.

        Raises:
            TypeError: If wrapped dataset does not return a dictionary sample.
        """
        with Stopwatch() as sw:
            item = self.dataset[index]
        if not isinstance(item, dict):
            raise TypeError(f"TimingWrapper expects dictionary samples, got {type(item)}")
        item[META_GETITEM_TIME] = sw.elapsed_seconds
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
