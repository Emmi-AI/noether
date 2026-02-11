#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

import torch

from noether.data.pipeline.sample_processor import SampleProcessor


class SupernodeSamplingSampleProcessor(SampleProcessor):
    """Randomly samples supernodes from a pointcloud."""

    def __init__(
        self,
        item: str,
        num_supernodes: int,
        supernode_idx_key: str = "supernode_idx",
        items_at_supernodes: set[str] | None = None,
        seed: int | None = None,
    ):
        """
        Args:
            item: Which key in the input_sample (i.e., pointcloud item) is used to sample supernodes.
            num_supernodes: How many supernodes to sample.
            items_at_supernodes: Selects items at the supernodes (e.g., pressure at supernodes). Defaults to None. These items are sampled accordingly and added to the output supernodes.
            seed: Random seed for deterministic sampling for evaluation. Default None (i.e., no seed). If not None,
                requires sample index to be present in batch.
        """

        self.item = item
        self.num_supernodes = num_supernodes
        self.supernode_idx_key = supernode_idx_key
        self.items_at_supernodes = items_at_supernodes
        self.seed = seed

    def __call__(self, input_sample: dict[str, Any]) -> dict[str, Any]:
        """Randomly samples supernodes from the pointcloud identified by `self.item`. The outer list and dicts are
        copied explicitly, the Any objects are not.

        Args:
            input_sample: Dictionary with the tensors of a single sample.

        Returns:
            Preprocessed copy of `input_sample` with supernodes sampled.
        """

        # copy to avoid changing method input
        output_sample = self.save_copy(input_sample)

        # sample supernodes
        cur_num_points = len(output_sample[self.item])
        generator: torch.Generator | None = None
        if self.seed is not None:
            if "index" not in output_sample:
                raise ValueError("Sample index is required for deterministic supernode sampling with a seed.")
            seed = output_sample["index"] + self.seed
            generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(cur_num_points, generator=generator)[: self.num_supernodes]

        # select items at supernode positions
        for item_at_supernodes in self.items_at_supernodes or []:
            item = output_sample[item_at_supernodes]
            data_at_supernode = item[perm]
            output_sample[f"supernode_{item_at_supernodes}"] = data_at_supernode
            output_sample[self.supernode_idx_key] = perm

        output_sample[self.supernode_idx_key] = perm

        return output_sample
