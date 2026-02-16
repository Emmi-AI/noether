#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections.abc import Sequence

import torch

from noether.data.pipeline.batch_processor import BatchProcessor


class PositionNormalizationBatchProcessor(BatchProcessor):
    """Post-processes data on a batch-level to normalize positions."""

    def __init__(
        self,
        items: list[str],
        raw_pos_min: Sequence[float],
        raw_pos_max: Sequence[float],
        scale: int | float = 1000,
    ):
        """

        Args:
            items: The position items (i.e., keys in the batch) to normalize.
            raw_pos_min: The minimum position in the source domain.
            raw_pos_max: The maximum position in the source domain.
            scale: The maximum value of the position. Defaults to 1000.
        """
        assert len(raw_pos_min) == len(raw_pos_max), "Raw position min and max must have the same length."

        self.items = items
        self.scale = scale
        self.raw_pos_min_tensor = torch.tensor(raw_pos_min).unsqueeze(0)
        self.raw_pos_max_tensor = torch.tensor(raw_pos_max).unsqueeze(0)
        self.raw_size = self.raw_pos_max_tensor - self.raw_pos_min_tensor

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """

        Args:
            batch: Collated batch.

        Return:
            Processed batch.
        """

        # copy to avoid changing method input
        batch = dict(batch)

        # process
        for item in self.items:
            if batch[item].ndim == self.raw_pos_min_tensor.ndim:
                # sparse tensor -> no additional dimension needed
                batch[item] = (batch[item] - self.raw_pos_min_tensor).div_(self.raw_size).mul_(self.scale)
            else:
                # dense tensor -> additional dimension needed
                assert batch[item].ndim == self.raw_pos_min_tensor.ndim + 1
                batch[item] = (
                    (batch[item] - self.raw_pos_min_tensor.unsqueeze(0))
                    .div_(self.raw_size.unsqueeze(0))
                    .mul_(self.scale)
                )

        return batch

    def denormalize(self, key: str, value: torch.Tensor) -> tuple[str, torch.Tensor]:
        """Inverts the normalization from the __call__ method of a single item in the batch.

        Args:
            key: The name of the item.
            value: The value of the item.

        Returns:
            (key, value): The same name and the denormalized value.
        """
        if key not in self.items:
            return key, value
        if value.ndim == self.raw_pos_min_tensor.ndim:
            # sparse tensor -> no additional dimension needed
            # fmt: off
            denormalized_value = (
                (value / self.scale)
                .mul_(self.raw_size.to(value.device))
                .add_(self.raw_pos_min_tensor.to(value.device))
            )
            # fmt: on
        else:
            # dense tensor -> additional dimension needed
            assert value.ndim == self.raw_pos_min_tensor.ndim + 1
            denormalized_value = (
                (value / self.scale)
                .mul_(self.raw_size.unsqueeze(0).to(value.device))
                .add_(self.raw_pos_min_tensor.unsqueeze(0).to(value.device))
            )
        return key, denormalized_value
