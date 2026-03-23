#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from abc import abstractmethod
from copy import deepcopy
from typing import Any, TypeVar

import torch

T = TypeVar("T")


class SampleProcessor:
    @abstractmethod
    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Pre-collate data on a sample-level. Each sample is a dictionary containing various data items per data point.
        Operations are performed on sample level, and are the same for every sample in the batch.
        Example:

            .. testcode::

                from noether.data.pipeline.sample_processor import SampleProcessor

                class MySampleProcessor(SampleProcessor):
                    def __call__(self, sample):
                        sample = self.save_copy(sample)  # Avoid modifying the original sample
                        sample["data"] = sample["data"] * 2
                        return sample

                sample_processor = MySampleProcessor()
                sample = {"data": torch.tensor([1, 2, 3]), "label": 0}
                pre_collated_sample = sample_processor(sample)
                print(pre_collated_sample["data"])

            .. testoutput::

                tensor([2, 4, 6])

        Args:
            sample: A samples of a batch.

        Return:
            Pre-collated sample.

        """

    def inverse(self, key: str, value: torch.Tensor) -> tuple[str, torch.Tensor]:
        """Inverts the transformation from the __call__ method of a single item in the batch. Only should be implemented if the SampleProcessor is invertable or if the identity function is valid.

        Args:
            key: The name of the item.
            value: The value of the item.

        Returns:
            (key, value): The (potentially) back-mapped name and the (potentially) denormalized value.
        """
        raise NotImplementedError("This method should be implemented by subclasses or this method is not invertable.")

    @staticmethod
    def save_copy(obj: T) -> T:
        """Make a deep copy of an object to avoid modifying the original object.

        Args:
            obj: Any object that should be copied.

        Returns:
            A deep copy of the input object.
        """
        # For PyTorch tensors, use clone() to avoid modifying the original tensor.
        if isinstance(obj, torch.Tensor):
            return obj.clone()  # type: ignore[return-value]

        # For objects with a built-in .copy() method (like NumPy arrays, lists, dicts).
        # This is often faster than a generic deepcopy.
        elif hasattr(obj, "copy") and callable(obj.copy):
            return obj.copy()  # type: ignore[no-any-return]

        # As a fallback for all other object types, perform a deep copy.
        else:
            return deepcopy(obj)
