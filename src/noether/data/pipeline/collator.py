#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import default_collate

CollatorType = Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]]


class Collator:
    """Base object that uses torch.utils.data.default_collate in its __call__ function. Derived classes can overwrite
    the __call__ implementation to implement a custom collate function. The collator can be passed to
    torch.utils.data.DataLoader via the collate_fn argument (DataLoader(dataset, batch_size=2, collate_fn=Collator()).

    Example:

        .. testcode::

            from noether.data.pipeline.collator import Collator

            collator = Collator()
            num_samples = 2
            samples = [{"data": torch.randn(3, 256, 256)} for _ in range(num_samples)]
            batch = collator(samples)
            print(batch["data"].shape)

        .. testoutput::

            torch.Size([2, 3, 256, 256])

    """

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collates a batch of samples.

        Args:
            batch: A list of samples to collate. Each sample is a dictionary containing tensors that need to be
                collated. For example, in an image classification task, a batch of size 2 would be
                [dict(image=torch.randn(3, 256, 256), label=0), dict(image=torch.randn(3, 256, 256), label=1).

        Returns:
            Collated batch of samples. The example image classification batch from above would be collated into
                dict(image=torch.randn(2, 3, 256, 256), label=torch.tensor([0, 1])).
        """
        return default_collate(batch)  # type: ignore[no-any-return]
