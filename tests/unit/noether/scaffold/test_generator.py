#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
import yaml

from noether.scaffold.choices import DatasetChoice, HardwareChoice, ModelChoice, OptimizerChoice, TrackerChoice
from noether.scaffold.config import resolve_config
from noether.scaffold.generator import generate_project

MODELS = list(ModelChoice)
DATASETS = list(DatasetChoice)
COMBOS = list(itertools.product(MODELS, DATASETS))


def _generate(tmp_path: Path, **overrides):
    """Helper to generate a project with sensible defaults, accepting overrides."""
    defaults = dict(
        project_name="test_proj",
        model=ModelChoice.UPT,
        dataset=DatasetChoice.SHAPENET_CAR,
        dataset_path="/tmp/fake_data",
        optimizer=OptimizerChoice.ADAMW,
        tracker=TrackerChoice.DISABLED,
        hardware=HardwareChoice.GPU,
        wandb_entity=None,
    )
    defaults.update(overrides)
    name = defaults["project_name"]
    proj = tmp_path / name
    config = resolve_config(**defaults, project_dir=proj)
    generate_project(config)
    return proj


@pytest.mark.parametrize(("model", "dataset"), COMBOS, ids=[f"{m.value}-{d.value}" for m, d in COMBOS])
def test_generate_project(tmp_path: Path, model: ModelChoice, dataset: DatasetChoice) -> None:
    proj = _generate(tmp_path, model=model, dataset=dataset)

    # Expected directories exist
    assert (proj / "callbacks").is_dir()
    assert (proj / "configs").is_dir()
    assert (proj / "model").is_dir()
    assert (proj / "pipeline").is_dir()
    assert (proj / "schemas").is_dir()
    assert (proj / "trainers").is_dir()

    # All YAML files parse without error
    for yf in proj.rglob("*.yaml"):
        content = yf.read_text()
        lines = [
            line for line in content.splitlines() if not line.startswith("# @package")
        ]  # remove Hydra directives to avoid YAML parsing issues
        yaml.safe_load("\n".join(lines))

    # No unresolved placeholders
    for ext in ("*.py", "*.yaml"):
        for f in proj.rglob(ext):
            content = f.read_text()
            for placeholder in ("__PROJECT__", "__CLASS__", "__DATASET_PATH__", "__OPTIMIZER__", "__TRACKER__"):
                assert placeholder not in content, f"Unresolved {placeholder} in {f.relative_to(proj)}"
