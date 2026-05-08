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


_INTEGRATION_DIR = "tests/integration/"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Per-item collection tweaks.

    1. Inject ``_stabilize_benchmark_environment`` into every
       ``@pytest.mark.benchmark`` test (prepending it ensures it is resolved
       before any function-scoped fixtures the benchmark relies on).
    2. Auto-apply ``@pytest.mark.integration`` to anything collected from
       ``tests/integration/`` so the suite can be filtered as
       ``pytest -m "not integration"`` (or run in isolation as
       ``pytest -m integration``) without per-file boilerplate.
    """
    rootpath = config.rootpath
    for item in items:
        if item.get_closest_marker("benchmark") is not None:
            fixturenames = getattr(item, "fixturenames", None)
            if fixturenames is not None and "_stabilize_benchmark_environment" not in fixturenames:
                fixturenames.insert(0, "_stabilize_benchmark_environment")

        try:
            rel = item.path.relative_to(rootpath).as_posix()
        except ValueError:
            continue
        if rel.startswith(_INTEGRATION_DIR) and item.get_closest_marker("integration") is None:
            item.add_marker(pytest.mark.integration)
