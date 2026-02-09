#  Copyright © 2025 Emmi AI GmbH. All rights reserved.
from __future__ import annotations

from typing import Any

import numpy as np

from noether.data.base import Dataset, DatasetWrapper


class PropertySubsetWrapper(DatasetWrapper):
    """Wrapper around arbitrary noether.data.Dataset instances to make __getitem__ load the properties that are defined
    in the `properties` attribute of this wrapper. For example, if we have a dataset that contains three kinds of items: "x", "y", and "z"
    (i.e., the dataset implements `getitem_x`, `getitem_y`, and `getitem_z` methods), we can create a PropertySubsetWrapper around that dataset with `properties={"x", "y"}`.
    to only load "x" and "y" when __getitem__ is called. This is useful to avoid loading unnecessary data from disk. For example, it might be that you need different items from the same dataset
    during training and validation. During training, you might only need "x" and "y", while during validation you might need "x", "y", and "z".
    By using a PropertySubsetWrapper, you can create two different datasets for training and validation that only load the necessary items.

    Example:

    .. code-block:: python

        from noether.data import PropertySubsetWrapper, Dataset


        class DummyDataset(Dataset):
            def __init__(self):
                self.data = torch.arange(10)

            def getitem_x(self, idx):
                return self.data[idx] * 2

            def getitem_y(self, idx):
                return self.data[idx] + 3

            def getitem_z(self, idx):
                return self.data[idx] - 5

            def __len__(self):
                return len(self.data)


        dataset = DummyDataset()
        wrapper = PropertySubsetWrapper(dataset=dataset, properties={"x", "y"})
        sample = wrapper[4]  # calls dataset.getitem_x(4) and dataset.getitem_y(4), getitem_z is not called
        sample  # {"x": 8, "y": 7}
        wrapper.properties  # {"x", "y"}

    """

    def __init__(self, dataset: Dataset, properties: set[str]):
        """

        Args:
            dataset: Base dataset to be wrapped.
            properties: Which properties to load from the wrapped dataset when __getitem__ is called.
        Raises:
            TypeError: If properties is not a set.
            ValueError: If properties is empty or if any property does not correspond to a getitem
        """
        super().__init__(dataset=dataset)
        if not isinstance(properties, set):
            raise TypeError("Properties must be a set.")
        self.properties = properties

        # split properties into _getitem_fns
        self._getitem_functions = {}
        if len(properties) == 0:
            raise ValueError("Properties must contain at least one item.")
        for prop in properties:
            if prop == "index":
                self._getitem_functions[prop] = self._getitem_index
            else:
                function_name = f"getitem_{prop}"
                # check that dataset implements getitem (wrappers can use the getitem of their child)
                if not hasattr(self.dataset, function_name):
                    raise AttributeError(
                        f"{type(self.dataset)} has no method {function_name}. "
                        "Make sure the dataset implements getitem_<mode> for all modes."
                    )
                self._getitem_functions[prop] = getattr(self.dataset, function_name)

    @staticmethod
    def _getitem_index(idx: int) -> int:
        """Generic implementation that allows "index" to be contained in mode without implementing getitem_index."""
        return idx

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Iterates over all items contained in self.mode and calls the getitem_<ITEM> method to load the specified
        property. The result of all getitem methods is returned as dictionary.

        Args:
            idx: Index of the sample that should be loaded.

        Returns:
            dict: Dictionary containing all retrieved data from the getitem methods.

        Raises:
            TypeError: If idx is not an integer.
            ValueError: If idx is negative.
            IndexError: If idx is out of bounds for the wrapped dataset.
        """
        if not isinstance(idx, int | np.int64):
            raise TypeError(f"Index must be an integer, got {type(idx)}.")
        if idx < 0:
            raise ValueError(f"Index must be non-negative, got {idx}.")
        if idx >= len(self.dataset):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self.dataset)}.")

        items: dict[str, Any] = {}
        for key, getitem_fn in self._getitem_functions.items():
            items[key] = getitem_fn(idx)
        return items

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
        if item == "properties":
            return self.properties
        if item == "__getitems__":
            # new torch versions (>=2.1) implements this which leads to wrappers being circumvented
            # -> disable batched getitems and call getitem instead
            # this occoured when doing DataLoader(dataset) where dataset is PropertySubsetWrapper(Subset(...))
            # Subset implements __getitems__ which leads to the fetcher from the DataLoader believing also the
            # PropertySubsetWrapper has a __getitems__ and therefore calls it instead of the __getitem__ function
            # returning None makes the DataLoader believe that __getitems__ is not supported
            return None
        return getattr(self.dataset, item)

    def __str__(self) -> str:
        dataset_str = (
            str(self.dataset.__class__.__name__) if self.dataset.__str__ is object.__str__ else str(self.dataset)
        )
        return f"{dataset_str} (properties={','.join(self.properties)})"

    @classmethod
    def from_included_excluded(
        cls, dataset: Dataset, included_properties: set[str] | None, excluded_properties: set[str] | None
    ) -> PropertySubsetWrapper | Dataset:
        """Creates a PropertySubsetWrapper from included and excluded properties.

        Args:
            dataset: Base dataset to be wrapped.
            included_properties: If defined, only these properties are included.
            excluded_properties: If defined, these properties are excluded.

        Returns:
            PropertySubsetWrapper: The created PropertySubsetWrapper.
        """
        available_properties = {
            method[len("getitem_") :]
            for method in dir(dataset)
            if method.startswith("getitem_") and callable(getattr(dataset, method))
        }
        available_properties.add("index")  # always allow index to be included

        if included_properties is not None and excluded_properties is not None:
            properties = included_properties - excluded_properties
        elif included_properties is not None:
            properties = included_properties
        elif excluded_properties is not None:
            properties = available_properties - excluded_properties
        else:
            return dataset  # no wrapping needed

        return cls(dataset=dataset, properties=properties)
