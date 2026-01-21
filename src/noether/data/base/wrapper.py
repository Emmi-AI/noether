#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from noether.data.base.dataset import Dataset


class DatasetWrapper:
    """Wrapper around arbitrary noether.data.Dataset instances to generically change something about the dataset.
    For example:
    - Create a subset of the dataset (noether.data.Subset)
    - Define which properties/items to load from the dataset, i.e., which getitem_* methods to call (noether.data.ModeWrapper)
    What exactly is changed depends on the specific implementation of the DatasetWrapper child class.
    """

    def __init__(self, dataset: Dataset | DatasetWrapper):
        """
        Args:
            dataset: base dataset to be wrapped
        """
        from noether.data.base.dataset import Dataset  # avoid circular import

        super().__init__()
        if not isinstance(dataset, (Dataset, DatasetWrapper)):
            raise ValueError(f"dataset must be of type noether.data.Dataset but got {type(dataset)}")
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, item: str) -> Any:
        """Wraps all getitem_ methods of the wrapped dataset such that they are called with the correct indices.
        Other properties are also directly passed-through to allow property access independent of whether
        the dataset is wrapped into a Subset or not. For example, if a classification dataset contains the property
        num_classes, wrapping that dataset into a subset would require dataset.dataset.num_classes instead of
        dataset.num_classes to access the field. This method makes sure that the Subset wrapper is fully transparent
        and the num_classes field can be accessed with dataset.num_classes.

        Args:
            item: name of the attribute to access (e.g., "getitem_x" to access dataset.getitem_x)

        Returns:
            Any: the result of the getitem_ (using the subset indices) or the attribute of the base dataset.
        """
        if item.startswith("getitem_"):
            # ctx is optional for getitem methods but wrappers should propagate it
            func = getattr(self.dataset, item)
            return partial(self._call_getitem, func)
        if item == "dataset":
            return getattr(super(), item)
        return getattr(self.dataset, item)

    def _call_getitem(self, func, idx, *args, **kwargs):
        """Wraps a getitem_ call to make sure only valid indices are accessed."""
        return func(idx, *args, **kwargs)

    def _get_all_getitem_names(self) -> list[str]:
        """Returns all names of getitem functions that are implemented. It will return the methods of the wrapped
        datasets as well as the methods of the wrapper if the wrapper implements additional getitem methods."""
        base_getitem_names = self.dataset.get_all_getitem_names()
        wrapper_getitem_names = [
            attr for attr in dir(self) if attr.startswith("getitem_") and callable(getattr(self, attr))
        ]
        return list(set(base_getitem_names + wrapper_getitem_names))

    def __dir__(self):
        """
        Combines the directory of the wrapper with the directory
        of the wrapped dataset.
        """
        # Get the wrapper's own attributes
        wrapper_dir = set(super().__dir__())

        # Get the wrapped dataset's attributes
        dataset_dir = set(dir(self.dataset))

        # Return the sorted union of both sets
        return sorted(wrapper_dir | dataset_dir)
