#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from abc import abstractmethod

import torch


class BatchProcessor:
    @abstractmethod
    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Processes data on a batch-level. The first dimension of each tensor in the batch is the batch dimension.
        Example:

            .. testcode::

                from noether.data.pipeline.batch_processor import BatchProcessor

                class MyBatchProcessor(BatchProcessor):
                    def __init__(self):
                        self.mean = torch.tensor([0.0, 0.0])
                        self.std = torch.tensor([1.0, 1.0])
                    def normalize(self, x):
                        return (x - self.mean) / self.std
                    def __call__(self, batch):
                        batch['x'] = self.normalize(batch['x'])
                        return batch
                    def denormalize(self, key, value):
                        return key, value

                postprocessor = MyBatchProcessor()
                batch = {"x": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
                processed_batch = postprocessor(batch)
                print(processed_batch['x'])

            .. testoutput::

                tensor([[1., 2.],
                        [3., 4.]])
        Args:
            batch: Collated batch.

        Return:
            Processed batch.
        """

    @abstractmethod
    def denormalize(self, key: str, value: torch.Tensor) -> tuple[str, torch.Tensor]:
        """Inverts the normalization from the __call__ method of a single item in the batch. If nothing needs to be
        done for the denormalization, this method should simply return the passed key/value.

        Args:
            key: The name of the item.
            value: The value of the item.

        Returns:
            (key, value): The (potentially) back-mapped name and the (potentially) denormalized value.
        """
