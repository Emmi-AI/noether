#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

import torch

from noether.data.pipeline import SampleProcessor


class ConcatTensorSampleProcessor(SampleProcessor):
    """Concatenates multiple tensors into a single tensor."""

    def __init__(self, items: list[str], target_key: str, dim: int = 0):
        """

        Args:
            items: A list of keys in the sample dict whose tensors should be concatenated.
            target_key: The key in the sample dict where the concatenated tensor will be stored.
            dim: The dimension along which to concatenate the tensors. Defaults to 0.
        """
        self.items = items
        self.target_key = target_key
        self.dim = dim

    def __call__(self, input_sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        """_summary_

        Args:
            sample: _description_

        Returns:
            _description_
        """
        output_sample = self.save_copy(input_sample)
        output_sample[self.target_key] = torch.concat([output_sample[item] for item in self.items], dim=self.dim)

        return output_sample
