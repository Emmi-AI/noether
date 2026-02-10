#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

import torch

from noether.data.pipeline.sample_processor import SampleProcessor


class PointSamplingSampleProcessor(SampleProcessor):
    """Randomly subsamples points from a tensor.


    .. code-block:: python

        # dummy example
        processor = PointSamplingSampleProcessor(
            items={"input_position", "output_position"},
            num_points=1024,
            seed=42,
        )
        input_sample = {
            "input_position": torch.randn(5000, 3),
            "output_position": torch.randn(5000, 3),
            "input_features": torch.randn(5000, 6),
        }
        output_sample = processor(input_sample)
        # output_sample['input_position'] will be a tensor of shape (1024, 3)
        # output_sample['output_position'] will be a tensor of shape (1024, 3)
        # output_sample['input_features'] will be unchanged.
        # If input features is also added to items, it will be of shape (1024, 6)
    """

    def __init__(self, items: set[str], num_points: int, seed: int | None = None):
        """

        Args:
            items: Which pointcloud items should be subsampled (e.g., input_position, output_position, ...). If multiple
            items are present, the subsampling will use identical indices for all items (e.g., to downsample
            output_position and output_pressure with the same subsampling).
            num_points: Number of points to sample.
            seed: Random seed for deterministic sampling for evaluation. Default None (i.e., no seed). If not None,
                requires sample index to be present in batch.
        """
        assert num_points > 0, "Number of points to sample must be positive."
        self.items = items
        self.num_points = num_points
        self.seed = seed

    def __call__(self, input_sample: dict[str, Any]) -> dict[str, Any]:
        """Subsamples the pointclouds identified by `self.items` with the same subsampling. The outer list and dicts
        are copied explicitly, the Any objects are not. However, the subsampled tensors are "copied" implicitly as
        sampling is implemented via random index access, which implicitly creates a copy of the underlying values.

        Args:
            input_sample: Dictionary with the tensors of a single sample.
        Returns:
            Preprocessed copy of `input_sample` with the specified items subsampled.
        """
        # copy to avoid changing method input
        output_sample = self.save_copy(input_sample)

        # apply preprocessing
        any_item = next(iter(self.items))

        # create perm
        first_item_tensor = output_sample[any_item]
        if not torch.is_tensor(first_item_tensor):
            raise ValueError(f"Item {any_item} in is not a tensor.")
        if self.seed is not None:
            if "index" not in output_sample:
                raise ValueError("Sample index is required for deterministic sampling with a seed.")
            seed = output_sample["index"] + self.seed
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None
        perm = torch.randperm(len(first_item_tensor), generator=generator)[: self.num_points]
        # subsample
        for item in self.items:
            tensor = output_sample[item]
            output_sample[item] = tensor[perm]

        return output_sample
