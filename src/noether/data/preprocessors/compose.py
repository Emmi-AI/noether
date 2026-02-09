#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from typing import Any

from noether.data.preprocessors.base import PreProcessor


class ComposePreProcess:
    """Compose multiple transforms and support inversion by reversing the sequence and inverting each transform.

    Example:

    .. code-block:: python

        from noether.data.preprocessors.compose import ComposePreProcess

        normalizer = ComposePreProcess(
            normalization_key="image",
            preprocessors=[
                MyPreProcessor1(),
                MyPreProcessor2(),
            ],
        )
        processed_data = normalizer(input_data)
        original_data = normalizer.inverse(processed_data)
    """

    def __init__(self, normalization_key: str, preprocessors: list[PreProcessor]) -> None:
        """

        Args:
            normalization_key: key to identify on which getitem_ in the dataset/tensor the preprocessor is applied.
            preprocessors: list of PreProcessor instances to compose.
        Raises:
            TypeError: If preprocessors is not a list or if any item in the list is not an instance of PreProcessor.
            ValueError: If the preprocessors list is empty.
        """

        super().__init__()
        if not isinstance(preprocessors, list):
            raise TypeError("preprocessors must be a list of PreProcessor instances.")
        for p in preprocessors:
            if not isinstance(p, PreProcessor):
                raise TypeError("All items in preprocessors must be instances of PreProcessor.")
        if len(preprocessors) == 0:
            raise ValueError("preprocessors list cannot be empty.")

        self.normalization_key = normalization_key
        self.transforms = preprocessors

    def __call__(self, x: Any) -> Any:
        """Apply each transform in sequence.
        Args:
            x: The input to be transformed.
        """
        for transform in self.transforms:
            x = transform(x)
        return x

    def inverse(self, x: Any) -> Any:
        """Return a transform that applies the inverse transformations in reverse order.
        Args:
            x: The input to be denormalized.
        """
        for t in reversed(self.transforms):
            x = t.denormalize(x)
        return x

    def __repr__(self) -> str:
        """Return a string representation of the composed transforms."""
        return f"ComposePreprocess(transforms={self.transforms})"

    def __len__(self) -> int:
        """Return the number of transforms."""
        return len(self.transforms)
