#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from pathlib import Path

import torch
from pydantic import Field

from noether.core.callbacks.base import CallBackBaseConfig
from noether.core.callbacks.periodic import PeriodicCallback
from noether.core.distributed.config import is_rank0
from noether.core.utils.training.counter import UpdateCounter


class PyTorchProfilerCallbackConfig(CallBackBaseConfig):
    """Configuration for the PyTorch profiler callback.

    The profiler uses ``torch.profiler.profile`` with a scheduled trace. Profiling is driven off of
    ``track_after_update_step`` hooks, i.e. the profiler is stepped once per optimizer update. The
    resulting traces are written to ``<run_output_path>/profiler`` and can be opened in
    TensorBoard or chrome://tracing.

    Recommended usage: limit training with ``trainer.max_updates`` to a value slightly larger than
    ``wait + warmup + active`` (times ``repeat`` if > 1).
    """

    kind: str | None = "aero_cfd.callbacks.PyTorchProfilerCallback"

    wait: int = Field(1, ge=0)
    """Number of steps to idle before warming up."""
    warmup: int = Field(1, ge=0)
    """Number of warmup steps (profiler runs but traces are discarded)."""
    active: int = Field(3, ge=1)
    """Number of active steps that are recorded in the trace."""
    repeat: int = Field(1, ge=0)
    """Number of times the (wait, warmup, active) cycle is repeated. 0 means repeat indefinitely."""

    record_shapes: bool = Field(True)
    """Whether to record input tensor shapes for each op."""
    profile_memory: bool = Field(False)
    """Whether to profile tensor memory usage (can add significant overhead)."""
    with_stack: bool = Field(False)
    """Whether to record Python call stacks for each op (can add significant overhead)."""
    with_flops: bool = Field(False)
    """Whether to record estimated FLOPs for each op."""
    with_modules: bool = Field(True)
    """Whether to record nn.Module hierarchy for each op."""

    profile_cuda: bool = Field(True)
    """Whether to profile CUDA operations. If False, only CPU operations are profiled."""
    profile_cpu: bool = Field(True)
    """Whether to profile CPU operations. If False, only CUDA operations are profiled."""

    trace_subdir: str = Field("profiler")
    """Subdirectory (relative to ``run_output_path``) where the trace files are written."""

    rank0_only: bool = Field(True)
    """If True, only rank 0 profiles (noop on other ranks). Avoids noisy/conflicting traces in
    multi-GPU runs."""


class PyTorchProfilerCallback(PeriodicCallback):
    """Profiles the training loop with :class:`torch.profiler.profile`.

    The profiler is entered in :meth:`before_training`, stepped once per optimizer update in
    :meth:`track_after_update_step`, and exited in :meth:`after_training`. Traces are written to
    ``<run_output_path>/<trace_subdir>`` via ``tensorboard_trace_handler`` and can be loaded in
    TensorBoard (``tensorboard --logdir <path>``) or inspected in ``chrome://tracing``.

    Note:
        ``every_n_updates=1`` must be set so that ``track_after_update_step`` is called on every
        update (any ``every_n_*`` value works — it only gates the unused ``periodic_callback``
        hook, not the tracking hooks).

    Example:
        .. code-block:: yaml

            callbacks:
            - kind: callbacks.PyTorchProfilerCallback
                every_n_updates: 1
                wait: 1
                warmup: 1
                active: 3
                repeat: 2
                record_shapes: true
                profile_memory: false
                with_stack: false
                with_flops: false
                with_modules: true
                activities:
                - cpu
                - cuda

    """

    def __init__(self, callback_config: PyTorchProfilerCallbackConfig, **kwargs):
        super().__init__(callback_config, **kwargs)

        self._config = callback_config
        self._profiler: torch.profiler.profile | None = None
        self._enabled = (not callback_config.rank0_only) or is_rank0()

    def _to_activities(self) -> list[torch.profiler.ProfilerActivity]:
        activities = []
        if self._config.profile_cpu:
            activities.append(torch.profiler.ProfilerActivity.CPU)
        if self._config.profile_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        return activities

    def before_training(self, *, update_counter: UpdateCounter) -> None:
        if not self._enabled:
            return

        trace_dir: Path = self.checkpoint_writer.path_provider.run_output_path / self._config.trace_subdir
        trace_dir.mkdir(parents=True, exist_ok=True)

        activities = self._to_activities()
        if torch.profiler.ProfilerActivity.CUDA in activities and not torch.cuda.is_available():
            self.logger.warning("CUDA activity requested but CUDA is not available — profiling CPU only")
            activities = [torch.profiler.ProfilerActivity.CPU]

        schedule = torch.profiler.schedule(
            wait=self._config.wait,
            warmup=self._config.warmup,
            active=self._config.active,
            repeat=self._config.repeat,
        )

        self._profiler = torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
            record_shapes=self._config.record_shapes,
            profile_memory=self._config.profile_memory,
            with_stack=self._config.with_stack,
            with_flops=self._config.with_flops,
            with_modules=self._config.with_modules,
        )
        self._profiler.__enter__()
        total_steps = (self._config.wait + self._config.warmup + self._config.active) * max(self._config.repeat, 1)
        self.logger.info(
            f"Started PyTorch profiler — traces will be written to {trace_dir} "
            f"(wait={self._config.wait}, warmup={self._config.warmup}, "
            f"active={self._config.active}, repeat={self._config.repeat}, "
            f"≈{total_steps} update steps needed)"
        )

    @torch.no_grad()
    def track_after_update_step(self, *, update_counter: UpdateCounter, times: dict[str, float]) -> None:
        if self._profiler is not None:
            self._profiler.step()

    def after_training(self, *, update_counter: UpdateCounter) -> None:
        if self._profiler is None:
            return
        self._profiler.__exit__(None, None, None)
        self._profiler = None
        self.logger.info("Stopped PyTorch profiler — traces flushed to disk")
