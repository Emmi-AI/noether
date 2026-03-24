#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .choices import HardwareChoice, TrackerChoice

TEMPLATES = Path(__file__).parent / "template_files"


@dataclass
class ScaffoldConfig:
    project_name: str
    tracker: TrackerChoice
    hardware: HardwareChoice
    project_dir: Path
    wandb_entity: str | None


def substitute(content: str, config: ScaffoldConfig) -> str:
    """Replace template placeholders with config values."""
    result = content.replace("__PROJECT__", config.project_name)
    result = result.replace("__TRACKER__", config.tracker.value)
    result = result.replace("__WANDB_ENTITY__", config.wandb_entity or "null")
    return result


def resolve_config(
    project_name: str,
    tracker: TrackerChoice,
    hardware: HardwareChoice,
    project_dir: Path,
    wandb_entity: str | None,
) -> ScaffoldConfig:
    """Build a fully-resolved ScaffoldConfig."""
    return ScaffoldConfig(
        project_name=project_name,
        tracker=tracker,
        hardware=hardware,
        project_dir=project_dir,
        wandb_entity=wandb_entity,
    )
