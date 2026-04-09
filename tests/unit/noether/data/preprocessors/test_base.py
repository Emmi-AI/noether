#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.data.preprocessors import PreProcessor


class ConcretePreProcessor(PreProcessor):
    """A concrete implementation of PreProcessor for testing purposes."""

    def __init__(self, normalization_key: str):
        super().__init__(normalization_key)

    def __call__(self, x):
        return x


class ScaleTenTimes(PreProcessor):
    """A concrete implementation of PreProcessor for testing purposes."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x * 10

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x / 10.0


class ScaleTwoTimes(PreProcessor):
    """A concrete implementation of PreProcessor for testing purposes."""

    def __call__(self, x):
        return x * 2


class TestPreProcessorInit:
    """Tests for the PreProcessor.__init__ method."""

    def test_init_success(self):
        """Tests successful initialization with valid arguments."""
        key = "test_key"
        preprocessor = ConcretePreProcessor(normalization_key=key)
        assert preprocessor.normalization_key == key

    def test_init_not_implemented(self):
        """Tests successful initialization with valid arguments."""
        key = "test_key"
        preprocessor = PreProcessor(normalization_key=key)
        assert preprocessor.normalization_key == key

        with pytest.raises(NotImplementedError):
            preprocessor(torch.tensor([1, 2, 3]))
        with pytest.raises(NotImplementedError):
            preprocessor.denormalize(torch.tensor([1, 2, 3]))

    def test_init_default_denormalizable(self):
        """Tests that denormalizable defaults to False."""
        key = "test_key"
        preprocessor = ConcretePreProcessor(normalization_key=key)
        assert preprocessor.normalization_key == key
        with pytest.raises(NotImplementedError):
            preprocessor.denormalize(torch.tensor([1, 2, 3]))

    @pytest.mark.parametrize("invalid_key", [123, None, [], (), {}])
    def test_init_invalid_normalization_key_type(self, invalid_key):
        """Tests that a TypeError is raised for a non-string normalization_key."""
        with pytest.raises(TypeError, match="normalization_key must be a string."):
            ConcretePreProcessor(normalization_key=invalid_key)

    def test_example_preprocessor(self):
        """Tests the example preprocessor."""
        preprocessor = ScaleTenTimes(normalization_key="test_key")
        input_tensor = torch.tensor([1, 2, 3])
        output_tensor = preprocessor(input_tensor)
        assert torch.equal(output_tensor, input_tensor * 10)

        denormalized_tensor = preprocessor.denormalize(output_tensor)
        assert torch.equal(denormalized_tensor, input_tensor)

    def test_example_preprocessor_denormalization(self):
        """Tests the example preprocessor denormalization."""
        preprocessor = ScaleTwoTimes(normalization_key="test_key")
        input_tensor = torch.tensor([1, 2, 3])
        output_tensor = preprocessor(input_tensor)
        assert torch.equal(output_tensor, input_tensor * 2)
        with pytest.raises(NotImplementedError):
            preprocessor.denormalize(output_tensor)
