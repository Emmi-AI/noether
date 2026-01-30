#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections.abc import Sequence
from typing import Any

import torch

from noether.data.pipeline.sample_processor import SampleProcessor


class SamplewiseNormalizationSampleProcessor(SampleProcessor):
    """Sample processor that normalizes samplewise to a specified range [low, high].
    However, normalization on a sample level should be done as part of the dataset.
    """

    def __init__(self, item: str, low: Sequence[float] | None, high: Sequence[float]):
        """

        Args:
            item: The item (i.e. key) in the input_sample to normalize.
            low: The low value of the range.
            high: The high value of the range.
        """

        self.item = item
        if low is None:
            self.low_tensor = None
        else:
            assert len(low) == len(high), "min and max must have the same length."
            self.low_tensor = torch.tensor(low).unsqueeze(0)
        self.high_tensor = torch.tensor(high).unsqueeze(0)

    def __call__(self, input_sample: dict[str, Any]) -> dict[str, Any]:
        """Pre-processes data on a sample-level to normalize samplewise to a specified range.

        Args:
            input_sample: Dictionary with the tensors of a single sample.

        Return:
           Preprocessed copy of `input_sample` with samplewise normalization applied.
        """
        # copy to avoid changing method input
        output_sample = self.save_copy(input_sample)
        # calculate the min and max of the sample
        sample_min = output_sample[self.item].min(dim=0).values
        sample_max = output_sample[self.item].max(dim=0).values

        if self.low_tensor is not None:
            # normalize the sample to the specified range
            output_sample[self.item] = (output_sample[self.item] - sample_min).div_(sample_max - sample_min)
            # scale to the specified range
            output_sample[self.item] = output_sample[self.item] * (self.high_tensor - self.low_tensor) + self.low_tensor
        else:
            # normalize the sample to the specified range
            output_sample[self.item] = (output_sample[self.item]).div_(sample_max)
            # scale to the specified range
            output_sample[self.item] = output_sample[self.item] * self.high_tensor

        return output_sample

    def inverse(self, key: str, value: torch.Tensor) -> tuple[str, torch.Tensor]:
        """Denormalization is not implemented."""
        raise NotImplementedError(
            f"{self.__class__.__name__} currently can't reconstruct the original scale. Either omit the "
            "sample processor in the dataset that should not use the normalization or add the min/max per-sample to the "
            "batch."
        )
