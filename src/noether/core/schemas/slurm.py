#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import getpass
import re
from typing import Any

from pydantic import BaseModel, field_validator, model_validator

# SLURM patterns that require a running job and cannot be resolved at submission time.
_UNSUPPORTED_FOLDER_PATTERNS: frozenset[str] = frozenset({"%j", "%J", "%A", "%a", "%N", "%x"})


class SlurmConfig(BaseModel):
    """Configuration for SLURM job submission via :mod:`submitit`.

    Field names mirror the keyword arguments accepted by
    :meth:`submitit.AutoExecutor.update_parameters`. All fields are optional
    and default to ``None``, meaning the cluster default is used.

    Note:
        Job stdout/stderr is owned by submitit and written to ``<folder>/<job_id>_log.out``
        / ``<folder>/<job_id>_log.err``. Use the ``folder`` field to control where these
        files land. SLURM ``--output``/``--error`` directives are intentionally not
        exposed; pass them via ``slurm_additional_parameters`` if you really need to
        override submitit's defaults (this disables ``job.stdout()`` helpers).
    """

    model_config = {"extra": "forbid"}

    # --- Executor constructor argument ---
    folder: str = "submitit_logs"
    """Directory where submitit writes the job script, pickled task, and stdout/stderr logs.
    Per-job files are named ``<job_id>_log.out`` etc. inside this directory.
    This is also used as the default ``output_path`` for training runs (see
    :attr:`ConfigSchema.output_path`).

    Supports ``%u`` (current username) interpolation, e.g.
    ``/home/%u/logs/experiment``. SLURM job-time patterns like ``%j`` are
    **not** supported because submitit needs the directory to exist before
    submission."""

    # --- AutoExecutor-generic parameters (mapped to update_parameters as-is) ---
    name: str | None = None
    """Job name (SLURM ``--job-name``)."""

    nodes: int | None = None
    """Number of nodes to allocate."""

    tasks_per_node: int | None = None
    """Number of tasks per allocated node."""

    cpus_per_task: int | None = None
    """Number of CPUs per task."""

    gpus_per_node: int | str | None = None
    """GPUs per node. Accepts a count or ``type:count`` (e.g. ``"a100:4"``)."""

    mem_gb: float | None = None
    """Memory per node in gigabytes."""

    timeout_min: int = 0
    """Wall-clock limit in minutes. Use 0 for no time limit"""

    stderr_to_stdout: bool | None = None
    """If True, merge stderr into stdout."""

    # --- Slurm-specific parameters (forwarded with the ``slurm_`` prefix) ---
    slurm_partition: str | None = None
    """Partition to submit the job to."""

    slurm_array_parallelism: int | None = None
    """Maximum number of array tasks running concurrently (SLURM ``%N`` in ``--array``)."""

    slurm_setup: list[str] | None = None
    """Shell commands run inside the job before the main command, e.g.
    ``["source .venv/bin/activate"]``."""

    slurm_additional_parameters: dict[str, Any] | None = None
    """Escape hatch for SLURM directives not exposed as first-class fields, e.g.
    ``{"nice": 0, "reservation": "my_res", "chdir": "/work"}``. Keys are passed as
    ``--key=value`` to ``sbatch``."""

    @field_validator("folder")
    @classmethod
    def _resolve_folder_patterns(cls, value: str) -> str:
        """Resolve ``%u`` to the current username; reject job-time SLURM patterns."""
        for pat in _UNSUPPORTED_FOLDER_PATTERNS:
            if pat in value:
                raise ValueError(
                    f"SLURM pattern '{pat}' in folder is not supported — submitit needs "
                    "the directory to exist before submission. Use %u (username) instead, "
                    "or remove the pattern."
                )
        return value.replace("%u", getpass.getuser())

    @field_validator("gpus_per_node")
    @classmethod
    def _validate_gpu_spec(cls, value: str | int | None) -> str | int | None:
        if value is None or isinstance(value, int):
            if isinstance(value, int) and value < 0:
                raise ValueError(f"gpus_per_node must be non-negative, got {value}.")
            return value
        if not re.match(r"^(\w+:)?\d+$", value):
            raise ValueError(
                f"Invalid gpus_per_node spec: '{value}'. Expected a count or 'type:count' (e.g. '2', 'a100:4')."
            )
        return value

    @model_validator(mode="after")
    def _set_tasks_per_node_if_gpus_set(self):
        """If gpus_per_node is set but tasks_per_node isn't, set tasks_per_node = gpus_per_node."""
        if self.gpus_per_node is not None and self.tasks_per_node is None:
            gpus = self.gpus_per_node if isinstance(self.gpus_per_node, int) else int(self.gpus_per_node.split(":")[-1])
            self.tasks_per_node = gpus
        return self

    def to_executor_kwargs(self) -> tuple[str, dict[str, Any]]:
        """Return ``(folder, update_parameters_kwargs)`` for :class:`submitit.AutoExecutor`.

        Generic fields are passed under their bare name; everything else keeps its
        ``slurm_`` prefix so submitit routes it to the slurm executor.

        Returns:
            A tuple ``(folder, kwargs)`` where ``folder`` is the executor's log directory
            and ``kwargs`` is the dict to splat into ``executor.update_parameters(**kwargs)``.
        """
        params: dict[str, Any] = {}
        for name, value in self:
            if name == "folder" or value is None:
                continue
            params[name] = value
        return self.folder, params
