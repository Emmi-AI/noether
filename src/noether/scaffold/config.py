#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import importlib.resources
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .choices import DatasetChoice, HardwareChoice, ModelChoice, OptimizerChoice, TrackerChoice

TEMPLATES = importlib.resources.files("noether.scaffold.template_files")


@dataclass
class ScaffoldConfig:
    project_name: str
    model: ModelChoice
    dataset: DatasetChoice
    dataset_path: str
    optimizer: OptimizerChoice
    tracker: TrackerChoice
    hardware: HardwareChoice
    project_dir: Path
    wandb_entity: str | None

    # Resolved from reference YAML
    reference: dict[str, Any] = field(default_factory=dict)


def substitute(content: str, config: ScaffoldConfig) -> str:
    """Replace template placeholders with config values."""
    result = content.replace("__PROJECT__", config.project_name)
    result = result.replace("__DATASET_PATH__", config.dataset_path)
    result = result.replace("__OPTIMIZER__", config.optimizer.value)
    result = result.replace("__TRACKER__", config.tracker.value)
    return result


def load_reference(dataset: DatasetChoice) -> dict[str, Any]:
    """Load reference YAML for a dataset from package resources."""
    ref_files = importlib.resources.files("noether.scaffold.references")
    ref_path = ref_files / f"{dataset.value}.yaml"
    with importlib.resources.as_file(ref_path) as p:
        return dict(yaml.safe_load(p.read_text()))


def resolve_config(
    project_name: str,
    model: ModelChoice,
    dataset: DatasetChoice,
    dataset_path: str,
    optimizer: OptimizerChoice,
    tracker: TrackerChoice,
    hardware: HardwareChoice,
    project_dir: Path,
    wandb_entity: str | None,
) -> ScaffoldConfig:
    """Build a fully-resolved ScaffoldConfig."""
    return ScaffoldConfig(
        project_name=project_name,
        model=model,
        dataset=dataset,
        dataset_path=dataset_path,
        optimizer=optimizer,
        tracker=tracker,
        hardware=hardware,
        project_dir=project_dir,
        wandb_entity=wandb_entity,
        reference=load_reference(dataset),
    )
