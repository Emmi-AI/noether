#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

import torch


class ConcatSparseTensorCollator:
    """Concatenates a sparse tensor along its first axis. Additionally, creates the batch_idx tensor which maps samples
    to their respective index in a batch.
    """

    def __init__(self, items: list[str], create_batch_idx: bool = False, batch_idx_key: str = "batch_idx"):
        """
        Args:
            items:  Which pointcloud items (i.e., keys in the sample dictionaries) should be collated.
            create_batch_idx: If true, creates a batch_idx tensor that maps samples to their index in the batch.
                Defaults to False.
            batch_idx_key: How the generated batch_idx tensor should be called. Defaults to "batch_idx". If multiple
                `batch_idx` tensors are generated, this needs to be set to distinguish between them. For example,
                if a subsampled pointcloud is loaded and also the raw pointcloud without subsampling one could use
                `batch_idx` for the subsampled pointcloud and `batch_idx_raw` for the raw pointcloud.
        """
        self.items = items
        self.create_batch_idx = create_batch_idx
        self.batch_idx_key = batch_idx_key

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Concatenates a sparse tensor along its first axis.

        Args:
            samples: List of individual samples retrieved from the dataset.

        Returns:
             Batched items produced by this collator, including a batch_idx tensor.
        """

        assert len(samples) > 0, "Cannot collate an empty list of samples."

        batch = {}
        # concatenate sparse tensors
        for item in self.items:
            batch[item] = torch.concat([samples[i][item] for i in range(len(samples))])
        # create batch_idx (e.g., collating supernode_idx doesn't need a batch_idx)
        if self.create_batch_idx:
            lens = [len(samples[i][self.items[0]]) for i in range(len(samples))]
            batch_idx = torch.empty(sum(lens), dtype=torch.long)
            start = 0
            cur_batch_idx = 0
            for i in range(len(lens)):
                end = start + lens[i]
                batch_idx[start:end] = cur_batch_idx
                start = end
                cur_batch_idx += 1
            assert self.batch_idx_key not in batch
            batch[self.batch_idx_key] = batch_idx
        return batch
