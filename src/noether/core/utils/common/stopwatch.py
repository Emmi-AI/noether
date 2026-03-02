#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import time
from types import TracebackType
from typing import Self

import torch


class Stopwatch:
    """A stopwatch class to measure elapsed time.

    Supports two timing backends depending on the provided device:

    * **CPU** (``device=None``): plain ``time.perf_counter()`` wall-clock timing.
    * **GPU** (CUDA or MPS ``device``): device event-based timing using
      ``torch.cuda.Event`` / ``torch.mps.Event`` for accurate GPU measurements.
      Events are recorded non-blocking in ``stop()`` / ``lap()`` and resolved
      lazily in ``elapsed_seconds``, which synchronizes each pending event pair
      individually before calling ``elapsed_time()``.

    Args:
        device: Optional device that selects the timing backend.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        self._use_gpu = device is not None and _is_gpu_device_available(device)
        self._running = False
        self._elapsed_seconds: list[float] = []

        # CPU timing state
        self._start_time: float | None = None
        self._lap_start_time: float | None = None

        # GPU event timing state
        self._device = device
        self._gpu_lap_start_event: torch.cuda.Event | torch.mps.Event | None = None
        # Pending (start, end) event pairs, resolved lazily in elapsed_seconds.
        self._gpu_pending_laps: list[tuple[torch.cuda.Event | torch.mps.Event, torch.cuda.Event | torch.mps.Event]] = []

    def _new_event(self) -> torch.cuda.Event | torch.mps.Event:
        assert self._device is not None
        if self._device.type == "cuda":
            return torch.cuda.Event(enable_timing=True)
        return torch.mps.Event(enable_timing=True)

    def start(self) -> Stopwatch:
        """Start the stopwatch."""
        assert not self._running, "can't start running stopwatch"
        self._running = True
        if self._use_gpu:
            self._gpu_lap_start_event = self._new_event()
            self._gpu_lap_start_event.record()
        else:
            self._start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        """Stop the stopwatch and return the elapsed time since the last lap.

        For GPU devices the return value is always ``float("nan")``; access
        ``elapsed_seconds`` after stopping to obtain the resolved time.
        """
        assert self._running, "can't stop a stopped stopwatch"
        if self._use_gpu:
            end_event = self._new_event()
            end_event.record()
            assert self._gpu_lap_start_event is not None
            self._gpu_pending_laps.append((self._gpu_lap_start_event, end_event))
            self._gpu_lap_start_event = None
            lap_time = float("nan")
        else:
            lap_start = self._lap_start_time if self._lap_start_time is not None else self._start_time
            assert lap_start is not None
            lap_time = time.perf_counter() - lap_start
            self._elapsed_seconds.append(lap_time)
            self._start_time = None
            self._lap_start_time = None
        self._running = False
        return lap_time

    def lap(self) -> float:
        """Record a lap time and return the elapsed time since the last lap.

        For GPU devices the return value is always ``float("nan")``; access
        ``elapsed_seconds`` after stopping to obtain the resolved time.
        """
        assert self._running, "lap requires stopwatch to be started"
        if self._use_gpu:
            end_event = self._new_event()
            end_event.record()
            assert self._gpu_lap_start_event is not None
            self._gpu_pending_laps.append((self._gpu_lap_start_event, end_event))
            self._gpu_lap_start_event = end_event
            lap_time = float("nan")
        else:
            lap_start = self._lap_start_time if self._lap_start_time is not None else self._start_time
            assert lap_start is not None
            lap_time = time.perf_counter() - lap_start
            self._elapsed_seconds.append(lap_time)
            self._lap_start_time = time.perf_counter()
        return lap_time

    @staticmethod
    def sync(device: torch.device) -> None:
        """Synchronize the given GPU device. No-op for CPU devices."""
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    def _flush_pending_gpu_laps(self) -> None:
        """Resolve pending GPU event pairs into elapsed seconds.

        Synchronizes each end event before calling ``elapsed_time()``.
        """
        for start_event, end_event in self._gpu_pending_laps:
            end_event.synchronize()
            self._elapsed_seconds.append(start_event.elapsed_time(end_event) / 1000.0)  # type: ignore[arg-type]
        self._gpu_pending_laps.clear()

    @property
    def last_lap_time(self) -> float:
        """Return the last lap time."""
        assert len(self._elapsed_seconds) > 0, "last_lap_time requires lap()/stop() to be called at least once"
        return self._elapsed_seconds[-1]

    @property
    def lap_count(self) -> int:
        """Return the number of laps recorded."""
        return len(self._elapsed_seconds)

    @property
    def average_lap_time(self) -> float:
        """Return the average lap time."""
        assert len(self._elapsed_seconds) > 0, "average_lap_time requires lap()/stop() to be called at least once"
        return sum(self._elapsed_seconds) / len(self._elapsed_seconds)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.stop()

    @property
    def elapsed_seconds(self) -> float:
        """Return the total elapsed time since the stopwatch was started.

        For GPU stopwatches, flushes any pending event pairs first,
        synchronizing each end event before calling ``elapsed_time()``.
        """
        assert not self._running, "elapsed_seconds requires stopwatch to be stopped"
        if self._gpu_pending_laps:
            self._flush_pending_gpu_laps()
        assert len(self._elapsed_seconds) > 0, "elapsed_seconds requires stopwatch to have been started and stopped"
        return sum(self._elapsed_seconds)

    @property
    def elapsed_milliseconds(self) -> float:
        """Return the total elapsed time since the stopwatch was started in milliseconds."""
        return self.elapsed_seconds * 1000


def _is_gpu_device_available(device: torch.device) -> bool:
    if device.type == "cuda":
        return torch.cuda.is_available()
    if device.type == "mps":
        return torch.backends.mps.is_available()
    return False
