#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import re

from pydantic import BaseModel, field_validator


class SlurmConfig(BaseModel):
    """Configuration for SLURM job submission via sbatch.

    All fields are optional and default to None, meaning the cluster default will be used.
    This schema covers all sbatch directives available in SLURM.
    """

    model_config = {"extra": "forbid"}

    # ---- Job Identification ----

    job_name: str | None = None
    """Name of the job (--job-name)."""

    # ---- Partition / Account / QoS ----

    partition: str | None = None
    """Partition to submit the job to (--partition). Multiple partitions can be comma-separated."""

    reservation: str | None = None
    """Reserve resources from a named reservation (--reservation)."""

    # ---- Node / Task Allocation ----

    nodes: int | str | None = None
    """Number of nodes to allocate (--nodes). Can be an integer or a range like '2-4'."""

    ntasks: int | None = None
    """Total number of tasks to launch (--ntasks)."""

    ntasks_per_node: int | None = None
    """Number of tasks per allocated node (--ntasks-per-node)."""

    cpus_per_task: int | None = None
    """Number of CPUs per task (--cpus-per-task)."""

    # ---- Memory ----

    mem: str | None = None
    """Memory per node (--mem), e.g. '4G', '512M', '0' for all available memory."""

    # ---- GPU / Accelerator Resources ----

    gpus: str | int | None = None
    """Total GPUs for the job (--gpus). Can be a count or 'type:count', e.g. 'v100:2'."""

    gpus_per_node: str | int | None = None
    """GPUs per node (--gpus-per-node). Can be a count or 'type:count', e.g. 'a100:4'."""

    gres: str | None = None
    """Generic consumable resources (--gres), e.g. 'gpu:2,shard:1'."""

    # ---- Time ----

    time: str | None = None
    """Wall clock time limit (--time). Formats: 'minutes', 'MM:SS', 'HH:MM:SS', 'D-HH', 'D-HH:MM', 'D-HH:MM:SS'."""

    begin: str | None = None
    """Defer job start until the specified time (--begin), e.g. '2024-01-15T10:00:00', 'now+1hour'."""

    # ---- Output ----

    output: str | None = None
    """File path for stdout (--output). Supports replacement symbols: %j (job ID), %x (job name), %A (array master ID), %a (array task ID), %N (node name), %u (user name)."""

    error: str | None = None
    """File path for stderr (--error). Same replacement symbols as output."""

    # ---- Job Arrays ----

    array: str | None = None
    """Job array specification (--array), e.g. '0-15', '1,3,5,7', '1-7%2' (max 2 concurrent)."""

    # ---- Dependencies ----

    kill_on_invalid_dep: bool | None = None
    """Kill the job if any dependency is invalid (--kill-on-invalid-dep)."""

    # ---- Scheduling Priority ----

    nice: int | None = None
    """Scheduling priority adjustment (--nice). Positive values lower priority."""

    # ---- Environment ----

    export: str | None = None
    """Environment variables to export to the job (--export). Use 'ALL' (default), 'NONE', or comma-separated list like 'VAR1,VAR2=value'."""

    chdir: str | None = None
    """Working directory for the job (--chdir)."""

    experiment_file: str | None = None
    """Path to experiment file for array jobs. Not an sbatch directive, but used in command generation."""

    source: str | None = None
    """Shell command to source before running the job (e.g. for activating a virtual environment)"""

    @field_validator("time")
    @classmethod
    def validate_time_format(cls, v: str | None) -> str | None:
        """Validate SLURM time format."""
        if v is None:
            return v
        if v.upper() in ("UNLIMITED", "INFINITE"):
            return v
        # Matches: minutes | MM:SS | HH:MM:SS | D-HH | D-HH:MM | D-HH:MM:SS
        if not re.match(r"^(\d+-)?(\d+:)?\d+(:\d+)?$", v):
            raise ValueError(
                f"Invalid SLURM time format: '{v}'. "
                "Expected: 'minutes', 'MM:SS', 'HH:MM:SS', 'D-HH', 'D-HH:MM', or 'D-HH:MM:SS'."
            )
        return v

    @field_validator("mem")
    @classmethod
    def validate_memory_format(cls, v: str | None) -> str | None:
        """Validate SLURM memory format (number with optional K/M/G/T suffix)."""
        if v is None:
            return v
        if not re.match(r"^\d+(\.\d+)?[KMGT]?B?$", v, re.IGNORECASE):
            raise ValueError(
                f"Invalid memory format: '{v}'. Expected a number with optional suffix K, M, G, or T (e.g. '4G', '512M')."
            )
        return v

    @field_validator("array")
    @classmethod
    def validate_array_format(cls, v: str | None) -> str | None:
        """Validate SLURM array specification."""
        if v is None:
            return v
        # Matches: ranges, lists, steps, and max concurrent (e.g. '0-15', '1,3,5', '1-7:2%4')
        if not re.match(r"^[\d,\-:]+(%\d+)?$", v):
            raise ValueError(f"Invalid array specification: '{v}'. Expected format like '0-15', '1,3,5,7', '1-7%2'.")
        return v

    @field_validator("gpus", "gpus_per_node")
    @classmethod
    def validate_gpu_spec(cls, v: str | int | None) -> str | int | None:
        """Validate GPU specification (count or type:count)."""
        if v is None:
            return v
        if isinstance(v, int):
            if v < 0:
                raise ValueError(f"GPU count must be non-negative, got {v}.")
            return v
        if not re.match(r"^(\w+:)?\d+$", v):
            raise ValueError(
                f"Invalid GPU specification: '{v}'. Expected a count or 'type:count' (e.g. '2', 'v100:4')."
            )
        return v
