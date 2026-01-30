#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from collections.abc import Sequence

import torch

from noether.data.pipeline.batch_processor import BatchProcessor


class MomentNormalizationBatchProcessor(BatchProcessor):
    """Normalizes a value with its mean and standard deviation (i.e., its moments) in a batch.

    Example:
    .. code-block:: python
        processor = MomentNormalizationBatchProcessor(
            items=['velocity', 'pressure'],
            mean=[1.0, 2.0],
            std=[0.1, 0.2],
        )
        batch = {
            'velocity': torch.tensor([[.., ..], [.., ..]]),
            'pressure': torch.tensor([[.., ..], [.., ..]]),
        }
        normalized_batch = processor(batch)
        # normalized_batch['velocity'] will be tensor([[.., ..], [.., ..]])
        # normalized_batch['pressure'] will be tensor([[.., ..], [.., ..]])
    """

    def __init__(self, items: list[str], mean: Sequence[float], std: Sequence[float]):
        """

        Args:
            items: the position items to normalize.
            mean: the mean of the value.
            std: the standard deviation of the value.
        """
        assert len(mean) == len(std), "Mean and standard deviation must have the same length."

        self.items = items
        self.mean_tensor = torch.tensor(mean).unsqueeze(0)
        self.std_tensor = torch.tensor(std).unsqueeze(0)

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Processes data on a batch-level to normalize a value to approximately mean=0 std=1.

        Args:
            batch: Collated batch.

        Return:
            Normalized batch.
        """

        # copy to avoid changing method input
        batch = dict(batch)

        # process
        for item in self.items:
            assert batch[item].ndim == self.mean_tensor.ndim
            batch[item] = (batch[item] - self.mean_tensor).div_(self.std_tensor)

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
        assert value.ndim == self.mean_tensor.ndim
        denormalized_value = value * self.std_tensor.to(value.device) + self.mean_tensor.to(value.device)
        return key, denormalized_value
