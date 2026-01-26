#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Literal

import torch
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, field_serializer, field_validator, model_validator

from noether.core.schemas.dataset import DatasetBaseConfig
from noether.core.schemas.models import ModelBaseConfig
from noether.core.schemas.normalizers import AnyNormalizer
from noether.core.schemas.trackers import WandBTrackerSchema
from noether.core.schemas.trainers import BaseTrainerConfig
from noether.core.utils.common import validate_path


class StaticConfigSchema(BaseModel):
    output_path: str
    """Path to store all outputs of the run, including logs, checkpoints, etc."""
    default_cudnn_benchmark: bool = True
    """Whether to enable cudnn benchmark mode by default."""
    default_cudnn_deterministic: bool | None = False
    """Whether to enable cudnn deterministic mode by default."""
    master_port: int | None = None
    """Port for distributed master node. If None, will be set from environment variable MASTER_PORT if available."""
    model_config = {"extra": "forbid"}  # Forbid extra fields in static configs

    @staticmethod
    def from_uri(uri: str) -> StaticConfigSchema:
        """Load static config from a given URI (file path)."""
        config_path = validate_path(uri, exists="must")
        raw_config = OmegaConf.load(config_path)
        resolved_config = OmegaConf.to_container(raw_config, resolve=True)
        return StaticConfigSchema(**resolved_config)  # type: ignore[arg-type]


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


class ConfigSchema(BaseModel):
    """Main configuration schema for experiments."""

    name: str | None = None
    """Name of the experiment."""
    accelerator: Literal["cpu", "gpu", "mps"] | None = None
    """Type of accelerator to use. Default is None, which lets the system choose the best available accelerator. GPU > MPS > CPU."""
    stage_name: str | None = None
    """Name of the current stage. I.e., train, finetune, test, etc"""
    dataset_kind: str | None = None
    """Kind of dataset to use i.e., class path."""
    dataset_root: str | None = None
    """Root directory of the dataset."""
    resume_run_id: str | None = None
    """Run ID to resume from. If None, start a new run. This can be used to resume training from the last checkpoint of a previous run when training was interrupted/failed."""
    resume_stage_name: str | None = None
    """Stage name to resume from. If None, resume from the default stage."""
    resume_checkpoint: str | None = None
    """Path to checkpoint to resume from. If None, start from scratch."""
    seed: int = Field(0)
    """Random seed for reproducibility."""
    dataset_statistics: dict[str, list[float | int]] | None = None
    """Pre-computed dataset statistics, e.g., mean and std for normalization. Since some tensors are multi-dimensional, the statistics are stored as lists."""
    dataset_normalizer: dict[str, list[AnyNormalizer]] | None = None
    """List of normalizers to apply to the dataset. The key is the data source name."""
    tracker: WandBTrackerSchema | None = None
    """Configuration for experiment tracking. If None, no tracking is used. If "disabled", tracking is explicitly disabled.  WandB is currently the only supported tracker."""
    run_id: str | None = None
    """Unique identifier for the run. If None, a new ID will be generated."""
    devices: str | None = None
    """Comma-separated list of device IDs to use. If None, all available devices will be used."""
    num_workers: int | None = None
    """Number of worker threads for data loading. If None,  will use (#CPUs / #GPUs - 1) workers"""
    cudnn_benchmark: bool = True
    """Whether to enable cudnn benchmark mode for this run."""
    cudnn_deterministic: bool = False
    """Whether to enable cudnn deterministic mode for this run."""

    datasets: dict[str, DatasetBaseConfig] = Field(...)

    model: ModelBaseConfig = Field(...)

    trainer: BaseTrainerConfig = Field(...)

    debug: bool = False
    """If True, enables debug mode with more verbose logging, no WandB logging and output written to debug directory."""
    store_code_in_output: bool = True
    """If True, store a copy of the current code in the output directory for reproducibility."""
    output_path: Path
    """Path to output directory."""
    master_port: int = Field(default_factory=master_port_from_env)
    """Port for distributed master node. If None, will be set from environment variable MASTER_PORT if available."""

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

    @field_validator("output_path", mode="after")
    @classmethod
    def validate_output_path(cls, value: Path) -> Path:
        """Validates that the output path is valid."""
        return validate_path(value, mkdir=True).absolute()

    @field_serializer("output_path", mode="plain")
    def serialize_output_path(self, value: Any) -> Any:
        return str(value)

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

    @model_validator(mode="after")
    def set_accelerator_if_unset(self) -> ConfigSchema:
        """Sets the accelerator if it is not already set."""
        if self.accelerator is None:
            if torch.cuda.is_available():
                self.accelerator = "gpu"
            elif torch.backends.mps.is_available():
                self.accelerator = "mps"
            else:
                self.accelerator = "cpu"
        return self
