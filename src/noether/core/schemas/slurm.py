#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import re

from pydantic import BaseModel, field_validator


class SlurmConfig(BaseModel):
    """Configuration for SLURM job submission via sbatch.

    All fields are optional and default to None, meaning the cluster default will be used.
    This schema covers all sbatch directives available in SLURM.
    """

    model_config = {"extra": "forbid"}

    job_name: str | None = None
    """Name of the job (--job-name)."""

    partition: str | None = None
    """Partition to submit the job to (--partition). Multiple partitions can be comma-separated."""

    reservation: str | None = None
    """Reserve resources from a named reservation (--reservation)."""

    nodes: int | str | None = None
    """Number of nodes to allocate (--nodes). Can be an integer or a range like '2-4'."""

    ntasks: int | None = None
    """Total number of tasks to launch (--ntasks)."""

    ntasks_per_node: int | None = None
    """Number of tasks per allocated node (--ntasks-per-node)."""

    cpus_per_task: int | None = None
    """Number of CPUs per task (--cpus-per-task)."""

    mem: str | None = None
    """Memory per node (--mem), e.g. '4G', '512M', '0' for all available memory."""

    gpus: str | int | None = None
    """Total GPUs for the job (--gpus). Can be a count or 'type:count', e.g. 'v100:2'."""

    gpus_per_node: str | int | None = None
    """GPUs per node (--gpus-per-node). Can be a count or 'type:count', e.g. 'a100:4'."""

    gres: str | None = None
    """Generic consumable resources (--gres), e.g. 'gpu:2,shard:1'."""

    time: str | None = None
    """Wall clock time limit (--time). Formats: 'minutes', 'MM:SS', 'HH:MM:SS', 'D-HH', 'D-HH:MM', 'D-HH:MM:SS'."""

    begin: str | None = None
    """Defer job start until the specified time (--begin), e.g. '2024-01-15T10:00:00', 'now+1hour'."""

    output: str | None = None
    """File path for stdout (--output). Supports replacement symbols: %j (job ID), %x (job name), %A (array master ID), %a (array task ID), %N (node name), %u (user name)."""

    error: str | None = None
    """File path for stderr (--error). Same replacement symbols as output."""

    array: str | None = None
    """Job array specification (--array), e.g. '0-15', '1,3,5,7', '1-7%2' (max 2 concurrent)."""

    kill_on_invalid_dep: bool | None = None
    """Kill the job if any dependency is invalid (--kill-on-invalid-dep)."""

    nice: int | None = None
    """Scheduling priority adjustment (--nice). Positive values lower priority."""
    chdir: str | None = None
    """Working directory for the job (--chdir)."""

    env_path: str | None = None
    """Shell command to source before running the job (e.g. for activating a virtual environment) which should be used as 'source env_path'."""

    # Fields that are not srun/sbatch directives
    _non_slurm_fields: frozenset[str] = frozenset({"env_path"})

    # Mapping from field names to srun flag names (where they differ from simple hyphenation)
    _flag_overrides: dict[str, str] = {
        "kill_on_invalid_dep": "kill-on-invalid-dep",
    }

    def to_srun_args(self) -> str:
        """Return a string of srun arguments for all non-None SLURM fields.

        Fields that are not actual srun directives (``experiment_file``, ``source``)
        are excluded. Boolean fields are rendered as bare flags when ``True`` and
        omitted when ``False``.
        """
        parts: list[str] = []
        for name, value in self:
            if value is None or name in self._non_slurm_fields:
                continue
            flag = f"--{self._flag_overrides.get(name, name.replace('_', '-'))}"
            if isinstance(value, bool):
                if value:
                    parts.append(flag)
            else:
                parts.append(f"{flag}={value}")
        return " ".join(parts)

    @field_validator("time")
    @classmethod
    def validate_time_format(cls, value: str | None) -> str | None:
        """Validate SLURM time format."""
        if value is None:
            return value
        if value.upper() in ("UNLIMITED", "INFINITE"):
            return value
        # Matches: minutes | MM:SS | HH:MM:SS | D-HH | D-HH:MM | D-HH:MM:SS
        if not re.match(r"^(\d+-)?(\d+:)?\d+(:\d+)?$", value):
            raise ValueError(
                f"Invalid SLURM time format: '{value}'. "
                "Expected: 'minutes', 'MM:SS', 'HH:MM:SS', 'D-HH', 'D-HH:MM', or 'D-HH:MM:SS'."
            )
        return value

    @field_validator("mem")
    @classmethod
    def validate_memory_format(cls, value: str | None) -> str | None:
        """Validate SLURM memory format (number with optional K/M/G/T suffix)."""
        if value is None:
            return value
        if not re.match(r"^\d+(\.\d+)?[KMGT]?B?$", value, re.IGNORECASE):
            raise ValueError(
                f"Invalid memory format: '{value}'. Expected a number with optional suffix K, M, G, or T (e.g. '4G', '512M')."
            )
        return value

    @field_validator("array")
    @classmethod
    def validate_array_format(cls, value: str | None) -> str | None:
        """Validate SLURM array specification."""
        if value is None:
            return value
        # Matches: ranges, lists, steps, and max concurrent (e.g. '0-15', '1,3,5', '1-7:2%4')
        if not re.match(r"^[\d,\-:]+(%\d+)?$", value):
            raise ValueError(
                f"Invalid array specification: '{value}'. Expected format like '0-15', '1,3,5,7', '1-7%2'."
            )
        return value

    @field_validator("gpus", "gpus_per_node")
    @classmethod
    def validate_gpu_spec(cls, value: str | int | None) -> str | int | None:
        """Validate GPU specification (count or type:count)."""
        if value is None:
            return value
        if isinstance(value, int):
            if value < 0:
                raise ValueError(f"GPU count must be non-negative, got {value}.")
            return value
        if not re.match(r"^(\w+:)?\d+$", value):
            raise ValueError(
                f"Invalid GPU specification: '{value}'. Expected a count or 'type:count' (e.g. '2', 'v100:4')."
            )
        return value
