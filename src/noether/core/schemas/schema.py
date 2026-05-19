#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

import torch
from pydantic import BaseModel, Field, field_serializer, field_validator, model_validator

from noether.core.schemas.dataset import DatasetBaseConfig
from noether.core.schemas.lib import Discriminated
from noether.core.schemas.models import ModelBaseConfig
from noether.core.schemas.slurm import SlurmConfig
from noether.core.schemas.trackers import BaseTrackerConfig
from noether.core.schemas.trainers import BaseTrainerConfig
from noether.core.utils.common import validate_path

ACCELERATOR_TYPES = Literal["cpu", "gpu", "mps"]


def master_port_from_env() -> int:
    """Gets the master port from the environment variable if available."""
    env_port = os.environ.get("MASTER_PORT")
    if env_port is not None:
        try:
            return int(env_port)
        except ValueError as e:
            raise ValueError(f"Environment variable MASTER_PORT='{env_port}' is not a valid integer") from e
    rand_gen = random.Random()
    return rand_gen.randint(20000, 60000)


def default_accelerator() -> ACCELERATOR_TYPES:
    """Sets the accelerator if it is not already set."""
    if torch.cuda.is_available():
        return "gpu"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


TModelConfig = TypeVar("TModelConfig", bound=ModelBaseConfig)
TDatasetConfig = TypeVar("TDatasetConfig", bound=DatasetBaseConfig)
TTrainerConfig = TypeVar("TTrainerConfig", bound=BaseTrainerConfig)


class ConfigSchema[TModelConfig: ModelBaseConfig, TDatasetConfig: DatasetBaseConfig, TTrainerConfig: BaseTrainerConfig](
    BaseModel
):
    """Root configuration schema for all experiments in Noether."""

    name: str | None = None
    """Name of the experiment."""
    accelerator: ACCELERATOR_TYPES = Field(default_factory=default_accelerator)
    """Type of accelerator to use. By default the system choose the best available accelerator. GPU > MPS > CPU."""
    stage_name: str | None = None
    """Name of the current stage. I.e., train, finetune, test, etc. When None, the run_id directory is used as output directory. Otherwise, run_id/stage_name is used."""
    dataset_kind: str | None = None
    """Kind of dataset to use i.e., class path."""
    dataset_root: str | None = None
    """Root directory of the dataset."""
    resume_run_id: str | None = None
    """Run ID to resume from. If None, start a new run. This can be used to resume training from the last checkpoint of a previous run when training was interrupted/failed."""
    resume_stage_name: str | None = None
    """Stage name to resume from. If None, resume from the default stage."""
    resume_checkpoint: str | None = None
    """Path to checkpoint to resume from. If None, the 'latest' checkpoint will be used."""
    resume_output_path: Path | None = None
    """Output root where the resume source run lives, when different from this run's ``output_path``. Set automatically by ``noether-eval`` (read from the source ``hp_resolved.yaml``) so that overriding ``output_path`` for the eval run does not break source-checkpoint lookup. Leave unset for ordinary training resume, which keeps writing alongside the original run."""
    seed: int = Field(0)
    """Random seed for reproducibility."""
    dataset_statistics: dict[str, list[float | int]] | None = None
    """Pre-computed dataset statistics, e.g., mean and std for normalization. Since some tensors are multi-dimensional, the statistics are stored as lists."""
    tracker: Annotated[BaseTrackerConfig, Discriminated(BaseTrackerConfig)] | None = Field(None)
    """Configuration for experiment tracking. If None, no tracking is used. If "disabled", tracking is explicitly disabled.  WandB is currently the only supported tracker."""
    run_id: str | None = None
    """Unique identifier for the run. When running under SLURM and not set explicitly,
    defaults to the SLURM job ID (e.g. ``12345_2`` for array task 2 of job 12345).
    Otherwise a new ID is generated at runtime."""
    devices: str | None = None
    """Comma-separated list of device IDs to use. If None, all available devices will be used."""
    num_workers: int | None = None
    """Number of worker threads for data loading. If None,  will use (#CPUs / #GPUs - 1) workers"""
    cudnn_benchmark: bool = True
    """Whether to enable cudnn benchmark mode for this run."""
    cudnn_deterministic: bool = False
    """Whether to enable cudnn deterministic mode for this run."""

    datasets: dict[str, Annotated[TDatasetConfig, Discriminated(DatasetBaseConfig)]] = Field(...)
    """Configuration for datasets. The key is the dataset and value is the configuration for that dataset.
    See :class:`~noether.core.schemas.dataset.DatasetBaseConfig` for available options.
    The key "train" is reserved for the training dataset, but if not provided, the first dataset will be used as training dataset by default,
    other keys are arbitrary and can be used to identify datasets for different stages, e.g., "train", "val", "test", etc. or different datasets for the same stage, e.g., "train_cfd", "train_wind_turbine", etc.
    """

    model: Annotated[TModelConfig, Discriminated(ModelBaseConfig)] = Field(...)
    """Configuration for the model. See :class:`~noether.core.schemas.models.ModelBaseConfig` for available options."""

    trainer: Annotated[TTrainerConfig, Discriminated(BaseTrainerConfig)] = Field(...)
    """Configuration for the trainer. See :class:`~noether.core.schemas.trainers.BaseTrainerConfig` for available options."""

    debug: bool = False
    """If True, enables debug mode with more verbose logging, no WandB logging and output written to debug directory."""
    store_code_in_output: bool = True
    """If True, store a copy of the current code in the output directory for reproducibility."""
    output_path: Path | None = None
    """Path to output directory. When omitted, defaults to ``slurm.folder`` if a
    ``slurm`` section is present. Raises a validation error when neither
    ``output_path`` nor ``slurm`` is provided."""
    overwrite_output: bool = False
    """Whether to overwrite the output directory if it already exists. Use with caution, as this will delete all existing outputs for the run_id/stage_name!"""
    master_port: int = Field(default_factory=master_port_from_env)
    """Port for distributed master node. If None, will be set from environment variable MASTER_PORT if available."""

    slurm: SlurmConfig | None = None
    """Configuration for SLURM job submission."""

    @field_validator("tracker", mode="before")
    @classmethod
    def empty_dict_is_none(cls, v: Any) -> Any:
        """Pre-processes tracker input before validation."""
        match v:
            # Case 1: Input is a string that case-insensitively matches "disabled"
            case str() as s if s.lower() == "disabled":
                return None
            # Case 2: Input is an empty dictionary
            case dict() as d if not d:
                return None
            # Case 3: All other inputs are passed through unchanged
            case _:
                return v

    @model_validator(mode="after")
    def _resolve_slurm_defaults(self) -> ConfigSchema:
        """Apply SLURM-aware defaults for ``output_path`` and ``run_id``.

        * ``output_path`` defaults to ``slurm.folder`` when omitted.
        * ``run_id`` defaults to the SLURM job ID when running inside a SLURM
          allocation (``SLURM_JOB_ID`` env var). For array jobs this becomes
          ``<array_job_id>_<task_id>``.
        """
        # --- output_path ---
        if self.output_path is None:
            if self.slurm is None:
                raise ValueError(
                    "output_path is required when no slurm section is present. "
                    "Either set output_path explicitly or add a slurm config (output_path defaults to slurm.folder)."
                )
            self.output_path = Path(self.slurm.folder)
        self.output_path = validate_path(self.output_path, mkdir=True).absolute()

        # --- run_id from SLURM env ---
        if self.run_id is None:
            slurm_job_id = os.environ.get("SLURM_JOB_ID")
            slurm_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
            slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
            if slurm_array_job_id is not None and slurm_array_task_id is not None:
                self.run_id = f"{slurm_array_job_id}_{slurm_array_task_id}"
            elif slurm_job_id is not None:
                self.run_id = slurm_job_id

        return self

    @field_serializer("output_path", mode="plain")
    def serialize_output_path(self, value: Any) -> str | None:
        return str(value) if value is not None else None

    @field_validator("master_port", mode="before")
    @classmethod
    def get_env_master_port(cls, value: Any) -> Any:
        """Sets master_port from environment variable if available."""
        if isinstance(value, list | tuple) and len(value) == 2 and all(isinstance(x, int) for x in value):
            low, high = value
            value = random.Random().randint(low, high)
        return value

    @property
    def config_schema_kind(self) -> str:
        """The fully qualified import path for the configuration class."""
        # Use __qualname__ to correctly handle nested classes
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"
