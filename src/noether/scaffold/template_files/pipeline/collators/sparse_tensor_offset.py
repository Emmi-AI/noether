#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from copy import deepcopy

import torch

from noether.data.pipeline.collator import Collator


class SparseTensorOffsetCollator(Collator):
    """Collates sparse tensors by concatenating them along the first axis and creating an offset tensor that maps
    each sample to its respective index in the batch.
    """

    def __init__(self, item: str, offset_key: str):
        self.item = item
        self.offset_key = offset_key

    def __call__(self, samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Concatenates sparse tensors along the first axis and creates an offset tensor.

        Args:
            samples: List of individual samples retrieved from the dataset.

        """
        offset = 0
        samples = [deepcopy(sample) for sample in samples]  # copy to avoid changing method input
        batch: dict[str, torch.Tensor] = {}
        for sample in samples:
            cur_num_points = len(sample[self.offset_key])
            sample[self.item] = sample[self.item] + offset
            offset += cur_num_points

        batch[self.item] = torch.concat([sample[self.item] for sample in samples])
        return batch
