#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import time
from unittest.mock import patch

import pytest
import torch

from noether.core.utils.common.stopwatch import Stopwatch

MODULE_PATH = "noether.core.utils.common.stopwatch"

_has_cuda = torch.cuda.is_available()
_has_mps = torch.backends.mps.is_available()

requires_cuda = pytest.mark.skipif(not _has_cuda, reason="CUDA not available")
requires_mps = pytest.mark.skipif(not _has_mps, reason="MPS not available")
requires_gpu = pytest.mark.skipif(not (_has_cuda or _has_mps), reason="No GPU available")


@patch(f"{MODULE_PATH}.time.perf_counter")
def test_stopwatch_basic_flow(mock_time):
    """Test start -> stop flow with controlled time."""
    # Define time points: Start at 0.0, Stop at 5.5
    mock_time.side_effect = [0.0, 5.5]

    sw = Stopwatch()
    sw.start()
    elapsed = sw.stop()

    assert elapsed == 5.5
    assert sw.elapsed_seconds == 5.5
    assert sw.elapsed_milliseconds == 5500.0
    assert sw.lap_count == 1


@patch(f"{MODULE_PATH}.time.perf_counter")
def test_stopwatch_laps(mock_time):
    """
    Test multiple laps.

    Sequence of calls to perf_counter:
    1. start() -> returns 10.0
    2. lap()   -> returns 12.0 (calc diff)
    3. lap()   -> returns 12.0 (reset lap start)
    4. stop()  -> returns 15.0

    Expected Math:
    Lap 1: 12.0 - 10.0 = 2.0
    Lap 2: 15.0 - 12.0 = 3.0
    Total: 5.0
    """
    mock_time.side_effect = [10.0, 12.0, 12.0, 15.0]

    sw = Stopwatch()
    sw.start()

    # First lap
    lap1 = sw.lap()
    assert lap1 == 2.0

    # Final stop (Second lap)
    lap2 = sw.stop()
    assert lap2 == 3.0

    assert sw.elapsed_seconds == 5.0
    assert sw.lap_count == 2
    assert sw.average_lap_time == 2.5
    assert sw.last_lap_time == 3.0


def test_context_manager_behavior():
    """Test that 'with' block starts and stops automatically."""
    # We patch the instance methods to verify they are called
    with patch.object(Stopwatch, "start") as mock_start, patch.object(Stopwatch, "stop") as mock_stop:
        with Stopwatch() as sw:
            pass

        mock_start.assert_called_once()
        mock_stop.assert_called_once()


def test_start_twice_fails():
    sw = Stopwatch()
    sw.start()
    with pytest.raises(AssertionError, match="can't start running stopwatch"):
        sw.start()


def test_stop_before_start_fails():
    sw = Stopwatch()
    with pytest.raises(AssertionError, match="can't stop a stopped stopwatch"):
        sw.stop()


def test_lap_before_start_fails():
    sw = Stopwatch()
    with pytest.raises(AssertionError, match="lap requires stopwatch to be started"):
        sw.lap()


def test_elapsed_seconds_while_running_fails():
    """elapsed_seconds should strictly only be accessible after stopping."""
    sw = Stopwatch()
    sw.start()
    with pytest.raises(AssertionError, match="elapsed_seconds requires stopwatch to be stopped"):
        _ = sw.elapsed_seconds


def test_properties_on_empty_stopwatch_fail():
    """Test properties that require at least one lap/stop."""
    sw = Stopwatch()
    # Not started yet
    with pytest.raises(AssertionError, match=r"requires lap\(\)/stop\(\)"):
        _ = sw.last_lap_time

    with pytest.raises(AssertionError, match=r"requires lap\(\)/stop\(\)"):
        _ = sw.average_lap_time

    with pytest.raises(AssertionError, match="requires stopwatch to have been started"):
        _ = sw.elapsed_seconds


def test_real_time_accuracy():
    """
    Sanity check using actual sleep.
    Uses pytest.approx because sleep is not 100% precise.
    """
    sw = Stopwatch()

    sw.start()
    time.sleep(0.1)
    lap1 = sw.lap()
    time.sleep(0.2)
    lap2 = sw.stop()

    # Check individual laps:
    assert lap1 == pytest.approx(0.1, abs=0.02)
    assert lap2 == pytest.approx(0.2, abs=0.02)

    # Check total:
    assert sw.elapsed_seconds == pytest.approx(0.3, abs=0.02)
    assert sw.lap_count == 2


def _gpu_device() -> torch.device:
    """Return the first available GPU device."""
    if _has_cuda:
        return torch.device("cuda")
    if _has_mps:
        return torch.device("mps")
    pytest.skip("No GPU available")


def _warmup_gpu(device: torch.device) -> None:
    """Run a dummy op to ensure the GPU device is initialized (MPS compiles shaders lazily)."""
    x = torch.randn(32, 32, device=device)
    _ = x @ x
    Stopwatch.sync(device)


@requires_gpu
def test_gpu_stopwatch_basic():
    """Start -> stop on GPU returns a positive elapsed time."""
    device = _gpu_device()
    _warmup_gpu(device)
    x = torch.randn(128, 128, device=device)
    with Stopwatch(device=device) as sw:
        _ = x @ x
    assert sw.elapsed_seconds >= 0
    assert sw.lap_count == 1


@requires_gpu
def test_gpu_stopwatch_multiple_steps():
    """GPU stopwatch works across many consecutive start/stop cycles without hanging."""
    device = _gpu_device()
    _warmup_gpu(device)
    x = torch.randn(128, 128, device=device)
    for i in range(20):
        with Stopwatch(device=device) as sw:
            _ = x @ x
        assert sw.elapsed_seconds >= 0, f"step {i} returned negative elapsed time"


@requires_cuda
def test_gpu_stopwatch_laps():
    """GPU stopwatch correctly records multiple laps.

    MPS does not support this because lap() reuses the end event as the start
    of the next lap, and torch.mps.Event.elapsed_time() rejects shared events.
    This is not an issue in practice since lap() is unused in the training loop.
    """
    device = torch.device("cuda")
    _warmup_gpu(device)
    x = torch.randn(128, 128, device=device)
    sw = Stopwatch(device=device)
    sw.start()
    _ = x @ x
    sw.lap()
    _ = x @ x
    sw.stop()
    # GPU laps are resolved lazily when elapsed_seconds is accessed
    assert sw.elapsed_seconds >= 0
    assert sw.lap_count == 2


@requires_gpu
def test_gpu_stopwatch_falls_back_to_cpu_for_unknown_device():
    """Stopwatch with a CPU device uses wall-clock timing, not GPU events."""
    sw = Stopwatch(device=torch.device("cpu"))
    assert not sw._use_gpu
    sw.start()
    time.sleep(0.05)
    sw.stop()
    assert sw.elapsed_seconds > 0


@requires_mps
def test_mps_event_sync_does_not_hang():
    """Regression test: per-event synchronization was replaced with device-level sync
    because torch.mps.Event.synchronize() can hang. Verify the flush path works."""
    device = torch.device("mps")
    x = torch.randn(256, 256, device=device)
    for _ in range(30):
        with Stopwatch(device=device) as sw:
            _ = x @ x
        # accessing elapsed_seconds triggers _flush_pending_gpu_laps
        assert sw.elapsed_seconds >= 0


@requires_cuda
def test_cuda_stopwatch_basic():
    """Smoke test: CUDA stopwatch records timing without error."""
    device = torch.device("cuda")
    x = torch.randn(256, 256, device=device)
    with Stopwatch(device=device) as sw:
        _ = x @ x
    assert sw.elapsed_seconds >= 0
