#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from noether.data.base.dataset import Dataset
from noether.data.base.wrapper import DatasetWrapper


class Subset(DatasetWrapper):
    """Wrapper around arbitrary noether.data.Dataset instances to only use a subset of the samples, similar to
    torch.utils.Subset, but with support for individual getitem_* methods instead of the __getitem__ method.

    Example:

    .. code-block:: python

        from noether.data import SubsetWrapper, Dataset

        len(dataset)  # 10
        subset = SubsetWrapper(dataset=dataset, indices=[0, 2, 5, 7])
        len(subset)  # 4
        subset[4]  # returns dataset[7]
    """

    def __init__(self, dataset: Dataset, indices: Sequence[int] | npt.NDArray[np.integer]):
        """Initializes the Subset wrapper.

        Args:
            dataset: The base dataset to be wrapped
            indices: valid indices of the wrapped dataset (list, tuple, or numpy array)
        """
        super().__init__(dataset=dataset)
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def _call_getitem(self, func: Callable, idx: int, ctx=None, *args, **kwargs) -> Any:
        """Wraps a getitem_ call to make sure only valid indices are accessed."""
        return func(int(self.indices[idx]), *args, **kwargs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Calls all implemented getitem functions and returns the results (using the remapped Subset indices instead
        of the original passed index).

        Returns:
            dict: dictionary of all getitem result

        """
        return self.dataset[self.indices[idx]]  # type: ignore[no-any-return]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """torch.utils.data.Dataset doesn't define __iter__ which makes 'for sample in dataset' run endlessly.

        Returns:
            Iterator[Dict[str, Any]]: an iterator of the type that would be returned by __getitem__
        """
        for i in range(len(self.indices)):
            yield self.dataset[self.indices[int(i)]]
