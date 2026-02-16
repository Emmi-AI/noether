#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

import torch

from noether.data.pipeline.sample_processor import SampleProcessor


class DropOutliersSampleProcessor(SampleProcessor):
    """Drops all outliers from key in a the input sample.

    .. code-block:: python

        # dummy example
        processor = DropOutliersSampleProcessor(
            item="measurement",
            affected_items={"related_measurement1", "related_measurement2"},
            min_value=0.0,
            max_value=100.0,
        )

        input_sample = {
            "measurement": torch.tensor([[10.0], [200.0], [-5.0], [50.0]]),
            "related_measurement1": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
            "related_measurement2": torch.tensor([[5.0], [6.0], [7.0], [8.0]]),
        }
        output_sample = processor(input_sample)
        # output_sample['measurement'] will be tensor([[10.0], [50.0]])
        # output_sample['related_measurement1'] will be tensor([[1.0], [4.0]])
        # output_sample['related_measurement2'] will be tensor([[5.0], [8.0]])
    """

    def __init__(
        self,
        item: str,
        affected_items: set[str] | None = None,
        min_value: float | None = None,
        max_value: float | None = None,
        min_quantile: float | None = None,
        max_quantile: float | None = None,
    ):
        """

        Args:
            item: The item to drop outliers from.
            affected_items: List of item (keys) that is also affected by outlier removal. Defaults to None.
            min_value: Drop outliers below min_value. Defaults to None.
            max_value: Drop outliers above max_value. Defaults to None.
            min_quantile: Drop outliers in/below min_quantile. Defaults to None.
            max_quantile: Drop outliers in/above max_value. Defaults to None.
        """
        self.item = item
        self.affected_items = affected_items
        self.min_value = min_value
        self.max_value = max_value
        self.min_quantile = min_quantile
        self.max_quantile = max_quantile

    @staticmethod
    def _combine_masks(mask1, mask2):
        if mask1 is None:
            return mask2
        if mask2 is None:
            return mask1
        return torch.logical_and(mask1, mask2)

    def __call__(self, input_sample: dict[str, Any]) -> dict[str, Any]:
        """

        Args:
            input_sample: Dictionary with the tensors of a single sample.

        Returns:
            Preprocessed copy of `input_sample` with outliers removed.
        """
        # copy to avoid changing method input
        output_sample = self.save_copy(input_sample)

        try:
            sample_item = output_sample[self.item]
        except KeyError:
            raise KeyError(
                f"Item '{self.item}' not found in sample. Available keys: {list(output_sample.keys())}"
            ) from None

        if not (output_sample[self.item].ndim == 2 and output_sample[self.item].size(1) == 1):
            raise ValueError(
                f"Expected item '{self.item}' to be a 2D tensor with a single column, "
                f"but got shape {output_sample[self.item].shape}."
            )
        sample_item = sample_item.squeeze(1)
        is_valid = None
        if self.min_value is not None:
            is_valid = self._combine_masks(sample_item >= self.min_value, is_valid)
        if self.max_value is not None:
            is_valid = self._combine_masks(sample_item <= self.max_value, is_valid)
        if self.min_quantile is not None:
            min_quantile = sample_item.quantile(q=self.min_quantile)
            is_valid = self._combine_masks(sample_item >= min_quantile, is_valid)
        if self.max_quantile is not None:
            max_quantile = sample_item.quantile(q=self.max_quantile)
            is_valid = self._combine_masks(sample_item <= max_quantile, is_valid)
        output_sample[self.item] = output_sample[self.item][is_valid]
        for affected_item in self.affected_items or []:
            try:
                output_sample[affected_item] = output_sample[affected_item][is_valid]
            except KeyError:
                raise KeyError(
                    f"Affected item '{affected_item}' not found in sample. Available keys: {list(output_sample.keys())}"
                ) from None
        return output_sample
