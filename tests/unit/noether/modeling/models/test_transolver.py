#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.core.schemas.models.transolver import TransolverConfig, TransolverPlusPlusConfig
from noether.modeling.models.transolver import Transolver


@pytest.fixture(params=[True, False], ids=["plusplus", "no_plusplus"])
def transolver_model(request):
    """Fixture to create a Transolver model instance."""
    use_transolver_plusplus = request.param

    config = (
        TransolverConfig(
            hidden_dim=3, num_heads=1, depth=2, attention_constructor="transolver", kind="single", name="transolver"
        )
        if not use_transolver_plusplus
        else TransolverPlusPlusConfig(
            hidden_dim=3,
            num_heads=1,
            depth=2,
            attention_constructor="transolver_plusplus",
            kind="single",
            name="transolver_plusplus",
        )
    )

    return Transolver(config)


def test_forward_valid_input(transolver_model):
    """Test the forward method with valid input."""
    batch_size = 4
    num_points = 10
    input_positions = torch.rand(batch_size, num_points, 3)  # Random normalized input

    output = transolver_model(input_positions, attn_kwargs={})

    assert isinstance(output, torch.Tensor), "Output should be a tensor."
    assert output.shape == (batch_size, num_points, 3), "Output shape is incorrect."


def test_forward_no_placeholder(transolver_model):
    """Test the forward method when placeholder is not used."""
    transolver_model.placeholder = None  # Disable placeholder
    batch_size = 3
    num_points = 7
    input_positions = torch.rand(batch_size, num_points, 3)  # Random input

    output = transolver_model(input_positions, attn_kwargs={})
    pressure_output = output

    assert pressure_output is not None, "Output should not be None even without placeholder."
