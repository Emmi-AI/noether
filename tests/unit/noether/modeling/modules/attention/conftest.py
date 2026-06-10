# Copyright © 2025 Emmi AI GmbH. All rights reserved.
import os
import pytest
import torch

# --- Device Fixtures ---
@pytest.fixture(scope="session")
def device():
    """Default device (CPU)."""
    return torch.device("cpu")

@pytest.fixture(scope="session")
def gpu_device():
    """GPU device, or skip if unavailable."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")

# --- Backend Parametrization ---
@pytest.fixture(
    scope="session",
    params=["sdpa", "flash_attn", "kernels/flash-attn3"],
    ids=["sdpa", "flash_attn", "kernels/flash-attn3"]
)
def attn_implementation(request):
    """Parametrize over attention backends."""
    return request.param

# --- Environment Setup for Backend ---
@pytest.fixture(scope="session", autouse=True)
def setup_attn_backend(request):
    """Set NOETHER_ATTN_IMPLEMENTATION for all tests in this session.
    Note: Uses session scope to avoid reloading modules repeatedly.
    """
    if "gpu" in request.fixturenames:
        # Only set backend for GPU tests
        backend = request.getfixturevalue("attn_implementation")
        os.environ["NOETHER_ATTN_IMPLEMENTATION"] = backend
        # Reload the flash attention module to pick up the new backend
        import importlib
        import noether.modeling.modules.attention._flash_attention as _fa
        importlib.reload(_fa)
    else:
        # Default to SDPA for CPU tests
        os.environ["NOETHER_ATTN_IMPLEMENTATION"] = "sdpa"