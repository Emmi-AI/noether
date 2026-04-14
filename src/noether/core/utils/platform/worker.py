#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import logging
import os
import subprocess
import sys
from functools import lru_cache

from noether.core.distributed import is_managed

logger = logging.getLogger(__name__)


def get_fair_cpu_count(reserve_for_main: int = 1) -> int:
    """Get the number of CPUs to use per device.

    If SLURM is used, the number of CPUs per task is used.
    Otherwise, CPUs are divided equally across devices (GPUs or a single CPU device).

    Args:
        reserve_for_main: CPUs to keep for the main / orchestration thread.

    Returns:
        Number of CPUs to use per device.
    """
    total_cpu_count = get_total_cpu_count()
    if total_cpu_count == 0:
        return 0

    device_count = _get_device_count()
    if device_count <= 0:
        logger.warning("No devices detected, assuming a single logical device.")
        device_count = 1

    if is_managed():
        cpus_per_task = _get_slurm_cpus_per_task(total_cpu_count)
        tasks_per_node = _get_int_env("SLURM_NTASKS_PER_NODE", 1)
        if cpus_per_task is not None:
            # currently only 1 GPU per task is supported
            if device_count != tasks_per_node:
                raise RuntimeError(f"Managed SLURM run expects one task per GPU ({device_count=}, {tasks_per_node=})")

            if device_count != tasks_per_node:
                logger.warning(
                    "Device count (%s) != SLURM_NTASKS_PER_NODE (%s)",
                    device_count,
                    tasks_per_node,
                )
            return max(cpus_per_task - reserve_for_main, 0)

    # Non-SLURM or no SLURM CPU info: divide evenly across devices
    return max(total_cpu_count // device_count - reserve_for_main, 0)


def _get_slurm_cpus_per_task(total_cpu_count: int) -> int | None:
    """Best-effort extraction of CPUs per task from SLURM env.

    Args:
        total_cpu_count: Number of CPUs to use per task.

    Returns:
        Number of CPUs per task or None in case of incomplete slurm env.
    """
    cpus_per_task = _get_int_env("SLURM_CPUS_PER_TASK")
    if cpus_per_task is not None:
        if cpus_per_task != total_cpu_count:
            logger.debug(
                "total_cpu_count (%s) != SLURM_CPUS_PER_TASK (%s)",
                total_cpu_count,
                cpus_per_task,
            )
        return cpus_per_task

    tasks_per_node = _get_int_env("SLURM_NTASKS_PER_NODE")
    cpus_on_node = _get_int_env("SLURM_CPUS_ON_NODE")
    if tasks_per_node and cpus_on_node:
        return cpus_on_node // tasks_per_node

    logger.warning("Incomplete SLURM CPU information; falling back to non-SLURM CPU distribution.")
    return None


def _get_int_env(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Environment variable %s=%r is not an int.", name, value)
        return default


@lru_cache(maxsize=1)
def _get_device_count() -> int:
    """Get number of devices on the current node (GPUs or MIG instances).

    - If MIG is enabled on a GPU, each MIG instance is counted as one device.
    - If MIG is not enabled on a GPU, the GPU itself is counted as one device.
    - If no GPU is found or nvidia-smi is missing, returns 1 (CPU-only logical device).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except OSError:
        logger.info("nvidia-smi not found; assuming CPU-only (1 logical device).")
        return 1

    lines = [l for l in result.stdout.strip().splitlines() if l]
    if not lines:
        logger.info("nvidia-smi -L returned no output; assuming CPU-only (1 logical device).")
        return 1

    devices_on_node = 0
    mig_count_for_gpu = 0
    seen_gpu = False

    for idx, line in enumerate(lines):
        is_gpu_line = line.startswith("GPU ")
        is_mig_line = "MIG" in line

        if is_mig_line:
            mig_count_for_gpu += 1

        # look ahead: new GPU section, or end of output
        next_is_gpu = idx + 1 < len(lines) and lines[idx + 1].startswith("GPU ")
        is_last = idx == len(lines) - 1

        if is_gpu_line:
            seen_gpu = True
            # If we hit a new GPU line but have accumulated MIGs from a previous GPU, flush them.
            if mig_count_for_gpu > 0:
                devices_on_node += mig_count_for_gpu
                mig_count_for_gpu = 0

        # If this is a GPU line that appears to be a "Plain GPU" (no MIGs follow it, or it's the last line), count it.
        if is_gpu_line and (next_is_gpu or is_last):
            devices_on_node += 1

        # Edge case: If the output ends with MIG lines, flush them.
        if is_last and not is_gpu_line and mig_count_for_gpu > 0:
            devices_on_node += mig_count_for_gpu

    if not seen_gpu or devices_on_node == 0:
        logger.warning("Could not parse 'nvidia-smi -L' output, assuming 1 device.")
        return 1

    return devices_on_node


@lru_cache(maxsize=1)
def get_total_cpu_count() -> int:
    """Get the total number of CPUs visible to this process."""
    if sys.version_info >= (3, 13) and hasattr(os, "process_cpu_count"):
        # New in Python 3.13
        return os.process_cpu_count() or 1
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        # Fallback for odd environments:
        cpu_count = os.cpu_count() or 1
        return cpu_count
