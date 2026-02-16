#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from abc import abstractmethod
from typing import Any

import numpy.typing as npt
import torch


class PreProcessor:
    """Base class for all data preprocessors.
    Example:

    .. code-block:: python

        class MyPreProcessor(PreProcessor):
            def __init__(self, normalization_key: "image"):
                super().__init__(normalization_key=normalization_key)

            def __call__(self, x):
                # Example processing: normalize to [0, 1]
                return x / 255.0

            def denormalize(self, x):
                # Example denormalization: scale back to [0, 255]
                return x * 255.0
    """

    def __init__(self, normalization_key: str):
        """

        Args:
            normalization_key: key to identify on which getitem_ in the dataset/tensor the preprocessor is applied.
        Raises:
            TypeError: If normalization_key is not a string.
        """
        if not isinstance(normalization_key, str):
            raise TypeError("normalization_key must be a string.")

        self.normalization_key = normalization_key

    @abstractmethod
    def __call__(self, x: Any) -> Any:
        """Process the input data and return the processed data.
        Args:
            x: The input data to process.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def denormalize(self, x: torch.Tensor) -> torch.Tensor | npt.NDArray:
        """Denormalizes the input data. This method should be overridden by subclasses if denormalization is supported.
        If denormalization is not supported, it raises NotImplementedError or decide to implement the identity function.

        Args:
            x: The input tensor to denormalize.

        """
        raise NotImplementedError("This preprocessor does not support denormalization.")
