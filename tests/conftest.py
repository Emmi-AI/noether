#  Copyright © 2026 Emmi AI GmbH. All rights reserved.
"""Project-wide pytest fixtures and hooks.

The ``_stabilize_benchmark_environment`` fixture below is automatically applied
to every test marked with ``@pytest.mark.benchmark`` (see the
:func:`pytest_collection_modifyitems` hook). This keeps benchmark-stability
setup — CPU thread pinning and GPU clock locking — in one place rather than
duplicated in each performance-test module.
"""

from __future__ import annotations

import os
import subprocess
import warnings
from collections.abc import Iterator

import pytest
import torch


@pytest.fixture(scope="session")
def _stabilize_benchmark_environment() -> Iterator[None]:
    """Stabilize clocks and thread counts for the duration of a benchmark session.

    - CPU: pin ``torch.set_num_threads(1)`` so CPU timings don't vary with intra-op parallelism.
    - CUDA: attempt ``nvidia-smi --lock-gpu-clocks`` at the max graphics clock reported by the
      driver (or ``NOETHER_BENCHMARK_GPU_CLOCK_MHZ`` when set). Locking usually requires
      privileged execution; on failure a warning is emitted and benchmarks continue with
      variable clocks (expect higher run-to-run variance).

    Clocks are reset and thread count restored on session teardown.

    Auto-applied to every ``@pytest.mark.benchmark`` test via
    :func:`pytest_collection_modifyitems`. Not ``autouse`` on purpose — non-benchmark
    tests should not pay the clock-locking cost or be forced single-threaded.
    """
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    locked = False
    if torch.cuda.is_available():
        env_mhz = os.environ.get("NOETHER_BENCHMARK_GPU_CLOCK_MHZ")
        if env_mhz:
            target_mhz = int(env_mhz)
        else:
            # ``clock_rate`` is the peak graphics clock in kHz.
            target_mhz = max(torch.cuda.get_device_properties(0).clock_rate // 1000, 1)

        try:
            result = subprocess.run(
                ["nvidia-smi", "--lock-gpu-clocks", f"{target_mhz},{target_mhz}"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                locked = True
            else:
                warnings.warn(
                    f"Could not lock GPU clocks to {target_mhz} MHz "
                    f"({(result.stderr or result.stdout).strip()}); "
                    "benchmarks will run with variable clocks.",
                    stacklevel=1,
                )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            warnings.warn(f"nvidia-smi unavailable for clock locking: {exc}", stacklevel=1)

    try:
        yield
    finally:
        torch.set_num_threads(prev_threads)
        if locked:
            subprocess.run(["nvidia-smi", "--reset-gpu-clocks"], capture_output=True, timeout=5, check=False)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Inject ``_stabilize_benchmark_environment`` into every ``@pytest.mark.benchmark`` test.

    Prepending the fixture to ``fixturenames`` ensures it is resolved before any
    function-scoped fixtures (including ``cache_flush``) that the benchmark relies on.
    """
    for item in items:
        if item.get_closest_marker("benchmark") is None:
            continue
        fixturenames = getattr(item, "fixturenames", None)
        if fixturenames is None or "_stabilize_benchmark_environment" in fixturenames:
            continue
        fixturenames.insert(0, "_stabilize_benchmark_environment")
