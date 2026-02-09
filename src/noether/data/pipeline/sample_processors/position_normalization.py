#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections.abc import Sequence
from typing import Any

import torch

from noether.data.pipeline.sample_processor import SampleProcessor


class PositionNormalizationSampleProcessor(SampleProcessor):
    """Pre-processes data on a sample-level to normalize positions.

    Should only be used when multiple items should be normalized with the same normalization.
    If only one item should be normalized, consider using the preprocessor
    :class:`~noether.data.preprocessors.normalizers.PositionNormalizer` instead.
    """

    def __init__(
        self,
        items: set[str],
        raw_pos_min: Sequence[float],
        raw_pos_max: Sequence[float],
        scale: int | float = 1000,
    ):
        """

        Args:
            items: The position items to normalize. I.e., keys of the input_sample dictionary that should be normalized.
            raw_pos_min: The minimum position in the source domain.
            raw_pos_max: The maximum position in the source domain.
            scale: The maximum value of the position. Defaults to 1000.
        """
        assert len(raw_pos_min) == len(raw_pos_max), "Raw position min and max must have the same length."

        self.items = items
        self.scale = scale
        self.raw_pos_min_tensor = torch.tensor(raw_pos_min)
        self.raw_pos_max_tensor = torch.tensor(raw_pos_max)
        self.raw_size = self.raw_pos_max_tensor - self.raw_pos_min_tensor

    def __call__(self, input_sample: dict[str, Any]) -> dict[str, Any]:
        """Pre-processes data on a sample-level to normalize positions.

        Args:
            input_sample: Dictionary with the tensors of a single sample.

        Return:
           Preprocessed copy of `input_sample` with positions normalized.
        """
        # copy to avoid changing method input
        output_sample = self.save_copy(input_sample)

        # process
        for item in self.items:
            output_sample[item] = (output_sample[item] - self.raw_pos_min_tensor).div_(self.raw_size).mul_(self.scale)

        return output_sample
