#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from pathlib import Path

from noether.scaffold.choices import HardwareChoice, TrackerChoice
from noether.scaffold.config import ScaffoldConfig, substitute


def test_substitute_replaces_all_placeholders() -> None:
    config = ScaffoldConfig(
        project_name="my_proj",
        tracker=TrackerChoice.WANDB,
        hardware=HardwareChoice.GPU,
        project_dir=Path("/tmp/my_proj"),
        wandb_entity="my-team",
    )
    template = "kind: __PROJECT__.model.Base\ntracker: __TRACKER__\nentity: __WANDB_ENTITY__"
    result = substitute(template, config)

    assert result == "kind: my_proj.model.Base\ntracker: wandb\nentity: my-team"


def test_substitute_wandb_entity_none_becomes_null() -> None:
    config = ScaffoldConfig(
        project_name="my_proj",
        tracker=TrackerChoice.DISABLED,
        hardware=HardwareChoice.GPU,
        project_dir=Path("/tmp/my_proj"),
        wandb_entity=None,
    )
    result = substitute("entity: __WANDB_ENTITY__", config)
    assert result == "entity: null"
