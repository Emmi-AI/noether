#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import torch

from noether.data.pipeline.batch_processor import BatchProcessor


class RenameKeysBatchProcessor(BatchProcessor):
    """Utility processor that simply renames the dictionary keys in a batch.
    Rename keys in the batch if they are in the key_map and keep old keys otherwise.
    Creates a new dictionary whose keys are renamed but uses references to the values of the old dict.
    This avoids copying the data and at the same time does not modify this function's input.

    """

    def __init__(self, key_map: dict[str, str]):
        """

        Args:
            key_map: dict with source keys as keys and target keys as values. The source keys are renamed target keys.
        """
        self.key_map = key_map

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Args:
            batch: The batch to rename the keys of.
        Returns:
            The batch with the keys renamed.
        """
        return {self.key_map.get(key, key): value for key, value in batch.items()}

    def denormalize(self, key: str, value: torch.Tensor) -> tuple[str, torch.Tensor]:
        """Inverts the key mapping from the __call__ method.

        Args:
            key: The name of the item.
            value: The value of the item.

        Returns:
            (key, value): The (potentially) remapped name and the unchanged value.
        """
        for source_key, target_key in self.key_map.items():
            if key == target_key:
                return source_key, value
        return key, value
