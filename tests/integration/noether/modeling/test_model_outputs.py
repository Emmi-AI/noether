#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from noether.core.factory import Factory
from noether.core.schemas.models import AnchorBranchedUPTConfig, TransformerConfig, TransolverConfig, UPTConfig
from noether.modeling.models.ab_upt import AnchoredBranchedUPT
from noether.modeling.models.transformer import Transformer
from noether.modeling.models.transolver import Transolver
from noether.modeling.models.upt import UPT
from noether.modeling.modules.attention import DotProductAttention, TransolverAttention


def test_transformer_determinism_regression_check(
    transformer_config: TransformerConfig,
) -> None:
    """
    Checks transformer output against a hardcoded expected sum to ensure that changes to the model are intentional.
    If the test fails, it indicates a change in the model's output which could be due to changes in architecture,
    initialization, or other factors affecting determinism.
    The expected_sum should be updated to the new value if the change is intentional.
    """
    torch.manual_seed(42)

    model = Factory().create(transformer_config)

    assert isinstance(model, Transformer)
    assert isinstance(model.blocks[0].attention_block, DotProductAttention)

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


def test_upt_determinism_regression_check(
    upt_config: UPTConfig,
    upt_input_generator: Callable[[int | None], dict[str, Any]],
) -> None:
    """
    Checks UPT output against a hardcoded expected sum to ensure that changes to the model are intentional.
    If the test fails, it indicates a change in the model's output which could be due to changes in architecture,
    initialization, or other factors affecting determinism.
    The expected_sum should be updated to the new value if the change is intentional.
    """
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


def test_ab_upt_determinism_regression_check(
    ab_upt_config: AnchorBranchedUPTConfig,
    ab_upt_input_generator: Callable[[int | None], dict[str, Any]],
) -> None:
    """
    Checks AB-UPT output against a hardcoded expected sum to ensure that changes to the model are intentional.
    If the test fails, it indicates a change in the model's output which could be due to changes in architecture,
    initialization, or other factors affecting determinism.
    The expected_sum should be updated to the new value if the change is intentional.
    """
    # 1. Set a random seed
    torch.manual_seed(42)

    model = Factory().create(ab_upt_config)

    assert isinstance(model, AnchoredBranchedUPT)

    # Generate inputs from pytest fixture
    inputs = ab_upt_input_generator(seed=42)

    with torch.no_grad():
        predictions, _ = model(**inputs)

    actual_sum = sum(tensor.sum().item() for tensor in predictions.values())
    print(f"AnchoredBranchedUPT determinism check: Output Sum = {actual_sum:.6f}")

    # Hardcoded expected sum from a previous run to check for regressions in determinism.
    expected_sum = 0.009243674110621214
    # Check that the actual sum is approximately equal to the expected sum within a small tolerance.
    assert actual_sum == pytest.approx(expected_sum, abs=1e-5)


def test_transolver_determinism_regression_check(
    transolver_config: TransolverConfig,
) -> None:
    """
    Checks Transolver output against a hardcoded expected sum to ensure that changes to the model are intentional.
    If the test fails, it indicates a change in the model's output which could be due to changes in architecture,
    initialization, or other factors affecting determinism.
    The expected_sum should be updated to the new value if the change is intentional.
    """
    torch.manual_seed(42)

    model = Factory().create(transolver_config)

    assert isinstance(model, Transolver)
    assert isinstance(model.blocks[0].attention_block, TransolverAttention)

    batch_size, seq_len = 2, 5
    sample = torch.randn(batch_size, seq_len, transolver_config.hidden_dim, generator=torch.Generator().manual_seed(42))

    with torch.no_grad():
        output = model(sample, attn_kwargs={})

    actual_sum = output.sum().item()
    print(f"Transolver determinism check: Output Sum = {actual_sum:.6f}")

    expected_sum = 5.363862

    assert actual_sum == pytest.approx(expected_sum, abs=1e-5)
