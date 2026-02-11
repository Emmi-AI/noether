#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

import torch

from noether.data.pipeline import SampleProcessor


class DefaultTensorSampleProcessor(SampleProcessor):
    """Create a tensor with a fixed dummy value, with a specified size.


    .. code-block:: python

        # dummy example
        processor = DefaultTensorSampleProcessor(
            item_key_name="default_tensor",
            feature_dim=128,
            size=10,
            default_value=0.5,
        )
        input_sample = {}
        output_sample = processor(input_sample)
        # output_sample['default_tensor'] will be a tensor of shape (10, 128) filled with 0.5
    """

    def __init__(
        self,
        item_key_name: str,
        feature_dim: int,
        size: int | None = None,
        matching_item_key: str | None = None,
        default_value: float = 0.0,
    ):
        """

        Args:
            item_key_name: key of the created default tensor in the output sample dict.
            default_value: value to fill the created default tensor with.
            feature_dim: size of the feature dimension of the created default tensor.
            size: size of the first dimension of the created default tensor.
            matching_item_key: key of an existing tensor in the input sample dict to match the size of the first dimension.
        """
        assert size is not None or matching_item_key is not None, (
            "size or matching_item_key must be specified. Otherwise size cannot be determined."
        )
        assert item_key_name is not None, "key_name must be specified."

        self.item_key_name = item_key_name
        self.feature_dim = feature_dim
        self.size = size
        self.matching_item_key = matching_item_key
        self.default_value = default_value

    def __call__(self, input_sample: dict[str, Any]) -> dict[str, Any]:
        """

        Args:
            input_sample: Dictionary with the tensors of a single sample.

        Returns:
            Preprocessed copy of `input_sample` with the specified default tensor created.
        """
        # copy to avoid changing method input

        output_sample = self.save_copy(input_sample)
        # mypy doesn't narrow types for instance attributes, so we need a local variable
        matching_item_key = self.matching_item_key
        if self.size is not None:
            dim = self.size
        elif matching_item_key is not None:
            dim = output_sample[matching_item_key].shape[0]
        else:
            raise ValueError("Either size or matching_item_key must be defined.")
        output_sample[self.item_key_name] = torch.empty(dim, self.feature_dim).fill_(self.default_value)

        return output_sample
