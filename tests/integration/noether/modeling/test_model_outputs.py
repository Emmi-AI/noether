#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from noether.core.factory import Factory
from noether.core.schemas.models import AnchorBranchedUPTConfig, TransformerConfig, UPTConfig
from noether.modeling.models.ab_upt import AnchoredBranchedUPT
from noether.modeling.models.transformer import Transformer
from noether.modeling.models.upt import UPT


# Check that the Transformer model produces the same output for the same input across multiple runs, ensuring determinism.
# If model architecture changes or initialization methods change, this test will fail
# and expected_sum will need to be updated to the new value.
def test_transformer_determinism_regression_check(
    transformer_config: TransformerConfig,
) -> None:
    torch.manual_seed(42)

    model = Factory().create(transformer_config)

    assert isinstance(model, Transformer)

    batch_size, seq_len = 2, 5
    sample = torch.randn(
        batch_size, seq_len, transformer_config.hidden_dim, generator=torch.Generator().manual_seed(42)
    )

    with torch.no_grad():
        output = model(sample, attn_kwargs={})

    actual_sum = output.sum().item()
    print(f"Transformer determinism check: Output Sum = {actual_sum:.6f}")

    # Hardcoded expected sum from a previous run to check for regressions in determinism.
    expected_sum = 5.218031
    # Check that the actual sum is approximately equal to the expected sum within a small tolerance.
    assert actual_sum == pytest.approx(expected_sum, abs=1e-5)


# Check that the UPT model produces the same output for the same input across multiple runs, ensuring determinism.
# If model architecture changes or initialization methods change, this test will fail
# and expected_sum will need to be updated to the new value.
def test_upt_determinism_regression_check(
    upt_config: UPTConfig,
    upt_input_generator: Callable[[int | None], dict[str, Any]],
) -> None:
    torch.manual_seed(42)

    model = Factory().create(upt_config)

    assert isinstance(model, UPT)

    # Generate inputs from pytest fixture
    inputs = upt_input_generator(seed=42)

    with torch.no_grad():
        output = model(**inputs)

    actual_sum = output.sum().item()
    print(f"UPT determinism check: Output Sum = {actual_sum:.6f}")

    # Hardcoded expected sum from a previous run to check for regressions in determinism.
    expected_sum = 0.067268
    # Check that the actual sum is approximately equal to the expected sum within a small tolerance.
    assert actual_sum == pytest.approx(expected_sum, abs=1e-5)


# Check that the AB-UPT model produces the same output for the same input across multiple runs, ensuring determinism.
# If model architecture changes or initialization methods change, this test will fail
# and expected_sum will need to be updated to the new value.
def test_ab_upt_determinism_regression_check(
    ab_upt_config: AnchorBranchedUPTConfig,
    ab_upt_input_generator: Callable[[int | None], dict[str, Any]],
) -> None:
    # 1. Set a random seed
    torch.manual_seed(42)

    model = Factory().create(ab_upt_config)

    assert isinstance(model, AnchoredBranchedUPT)

    # Generate inputs from pytest fixture
    inputs = ab_upt_input_generator(seed=42)

    with torch.no_grad():
        predictions = model(**inputs)

    actual_sum = sum(tensor.sum().item() for tensor in predictions.values())
    print(f"AnchoredBranchedUPT determinism check: Output Sum = {actual_sum:.6f}")

    # Hardcoded expected sum from a previous run to check for regressions in determinism.
    expected_sum = -0.021468846505740657
    # Check that the actual sum is approximately equal to the expected sum within a small tolerance.
    assert actual_sum == pytest.approx(expected_sum, abs=1e-5)
