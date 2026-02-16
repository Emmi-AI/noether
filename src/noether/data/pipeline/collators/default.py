#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

import torch
from torch.utils.data import default_collate


class DefaultCollator:
    """Applies `torch.utils.data.default_collate` to the specified items."""

    def __init__(self, items: list[str] | None = None, optional_items: list[str] | None = None):
        """

        Args:
            items: Items (i.e, keys) to apply `default_collate` to.
            optional_items: Items (i.e, keys) to apply `default_collate` to if they are present.
        """
        self.items = items
        self.optional_items = optional_items

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Applies `torch.utils.data.default_collate` to the specified items.

        Args:
            samples: List of individual samples retrieved from the dataset.

        Returns:
             Batched items produced by this collator.
        """
        batch = {}
        for item in self.items or []:
            batch[item] = default_collate([samples[i][item] for i in range(len(samples))])
        for item in self.optional_items or []:
            if item in samples[0]:
                batch[item] = default_collate([samples[i][item] for i in range(len(samples))])
        return batch
