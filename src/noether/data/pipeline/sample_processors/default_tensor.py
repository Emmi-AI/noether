#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

import torch

from noether.data.pipeline import SampleProcessor


class DefaultTensorSampleProcessor(SampleProcessor):
    """Create a tensor with a fixed dummy value, with a specified size."""

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
            item_key_name: _description_
            default_value: _description_
            feature_dim: _description_
            size: _description_
            matching_item_key: _description_
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
        """_summary_

        Args:
            input_sample: _description_

        Returns:
            _description_
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
