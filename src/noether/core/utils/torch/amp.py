#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
from contextlib import contextmanager
from typing import Any

import torch
from torch.amp.grad_scaler import GradScaler

FLOAT32_ALIASES = ["float32", "fp32"]
FLOAT16_ALIASES = ["float16", "fp16"]
BFLOAT16_ALIASES = ["bfloat16", "bf16"]
VALID_PRECISIONS = FLOAT32_ALIASES + FLOAT16_ALIASES + BFLOAT16_ALIASES

logger = logging.getLogger(__name__)


def get_supported_precision(
    desired_precision: str,
    device: torch.device,
) -> torch.dtype:
    """Returns desired_precision if it is supported and backup_precision otherwise. For example, bfloat16 is not
    supported by all GPUs.

    Args:
        desired_precision: The desired precision format.
        device: The selected device (e.g., torch.device("cuda")).

    Returns:
        torch.dtype: The most suitable precision supported by the device.
    """
    assert desired_precision in VALID_PRECISIONS
    if desired_precision in FLOAT32_ALIASES:
        return torch.float32
    if desired_precision in FLOAT16_ALIASES:
        desired_precision = "float16"
    if desired_precision in BFLOAT16_ALIASES:
        desired_precision = "bfloat16"

    if desired_precision == "bfloat16":
        if is_bfloat16_compatible(device):
            return torch.bfloat16
        else:
            raise RuntimeError("bfloat16 not supported on this device")

    if desired_precision == "float16":
        if is_float16_compatible(device):
            return torch.float16
        else:
            raise RuntimeError("float16 not supported on this device")

    logger.info("float16/bfloat16 not supported -> using float32")
    return torch.float32


def is_compatible(device: torch.device, dtype: torch.dtype) -> bool:
    """Checks if a given dtype is supported on a device.

    Args:
        device: The device to check compatibility.
        dtype: The data type to check.

    Returns:
        bool: True if the dtype is supported, False otherwise.
    """
    try:
        with torch.autocast(device_type=str(device), dtype=dtype):
            pass
    except RuntimeError:
        return False
    return True


def is_bfloat16_compatible(device: torch.device) -> bool:
    """Checks if bfloat16 precision is supported on the given device.

    Args:
        device: The device to check.

    Returns:
        bool: True if bfloat16 is supported, False otherwise.
    """
    return is_compatible(device, torch.bfloat16)


def is_float16_compatible(device: torch.device) -> bool:
    """Checks if float16 precision is supported on the given device.

    Args:
        device: The device to check.

    Returns:
        bool: True if float16 is supported, False otherwise.
    """
    return is_compatible(device, torch.float16)


class NoopContext:
    """A no-operation context manager that does nothing."""

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args, **kwargs) -> None:
        pass


class NoopGradScaler(GradScaler):
    """A no-operation gradient scaler that performs no scaling."""

    def __init__(self) -> None:
        super().__init__(enabled=False)

    def scale(self, outputs: Any) -> Any:
        return outputs

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        pass

    @staticmethod
    def step(optimizer: torch.optim.Optimizer, *args, **kwargs) -> None:
        optimizer.step(*args, **kwargs)

    def update(self, new_scale: float | torch.Tensor | None = None) -> None:
        pass


def get_grad_scaler_and_autocast_context(
    precision: torch.dtype,
    device: torch.device,
) -> tuple[GradScaler, torch.autocast | NoopContext]:
    """Returns the appropriate gradient scaler and autocast context manager for the given precision.

    Args:
        precision (torch.dtype): The desired precision.
        device (torch.device): The device where computation occurs.

    Returns:
        The corresponding scaler and autocast context.
    """
    if precision == torch.float32:
        return NoopGradScaler(), NoopContext()
    if precision == torch.bfloat16:
        # GradScaler shouldn't be necessary (https://github.com/pytorch/pytorch/issues/36169)
        return NoopGradScaler(), torch.autocast(str(device), dtype=precision)
    if precision == torch.float16:
        if str(device) == "cpu":
            return NoopGradScaler(), torch.autocast(str(device), dtype=precision)
        return GradScaler(device=device.type), torch.autocast(str(device), dtype=precision)
    raise NotImplementedError("Unsupported precision type")


@contextmanager
def disable(device_type: str):
    """Disables AMP for the given device.

    Args:
        device_type: The device type to disable AMP for.
    """
    if device_type == "mps":
        yield
    else:
        with torch.autocast(device_type=device_type, enabled=False):
            yield
