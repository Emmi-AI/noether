#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from noether.scaffold.choices import HardwareChoice, TrackerChoice
from noether.scaffold.config import resolve_config
from noether.scaffold.file_manager import FileManager


def _generate(tmp_path: Path, **overrides):
    """Helper to generate a project with sensible defaults, accepting overrides."""
    defaults = dict(
        project_name="test_proj",
        tracker=TrackerChoice.DISABLED,
        hardware=HardwareChoice.GPU,
        wandb_entity=None,
    )
    defaults.update(overrides)
    name = defaults["project_name"]
    proj = tmp_path / name
    config = resolve_config(**defaults, project_dir=proj)
    FileManager.copy_template_tree(config)
    return proj


def test_generate_project(tmp_path: Path) -> None:
    proj = _generate(tmp_path)
    pkg = proj / proj.name

    # pyproject.toml at project root with substituted project name
    assert (proj / "pyproject.toml").is_file()
    pyproject = (proj / "pyproject.toml").read_text()
    assert "test_proj" in pyproject
    assert "__PROJECT__" not in pyproject

    # Expected directories exist inside the package
    assert (pkg / "callbacks").is_dir()
    assert (pkg / "configs").is_dir()
    assert (pkg / "models").is_dir()
    assert (pkg / "pipelines").is_dir()
    assert (pkg / "schemas").is_dir()
    assert (pkg / "trainer").is_dir()
    assert (pkg / "datasets").is_dir()

    # All YAML files parse without error
    for yf in pkg.rglob("*.yaml"):
        content = yf.read_text()
        lines = [
            line for line in content.splitlines() if not line.startswith("# @package")
        ]  # remove Hydra directives to avoid YAML parsing issues
        yaml.safe_load("\n".join(lines))

    # Check all placeholders are substituted in all files
    for f in proj.rglob("*"):
        if not f.is_file():
            continue
        content = f.read_text()
        for placeholder in ("__PROJECT__", "__TRACKER__", "__WANDB_ENTITY__"):
            assert placeholder not in content, f"Unresolved {placeholder} in {f.relative_to(proj)}"


@pytest.mark.parametrize("tracker", list(TrackerChoice), ids=[t.value for t in TrackerChoice])
def test_tracker_choice(tmp_path: Path, tracker: TrackerChoice) -> None:
    proj = _generate(tmp_path, tracker=tracker)
    experiment = (proj / proj.name / "configs" / "base_experiment.yaml").read_text()
    assert f"- tracker: {tracker.value}" in experiment


def test_gpu_hardware_no_accelerator(tmp_path: Path) -> None:
    proj = _generate(tmp_path, hardware=HardwareChoice.GPU)
    experiment = (proj / proj.name / "configs" / "base_experiment.yaml").read_text()
    assert "accelerator:" not in experiment


@pytest.mark.parametrize("hardware", [HardwareChoice.MPS, HardwareChoice.CPU], ids=["mps", "cpu"])
def test_hardware_accelerator_appended(tmp_path: Path, hardware: HardwareChoice) -> None:
    proj = _generate(tmp_path, hardware=hardware)
    experiment = (proj / proj.name / "configs" / "base_experiment.yaml").read_text()
    assert f"accelerator: {hardware.value}" in experiment


def test_wandb_entity_substituted(tmp_path: Path) -> None:
    proj = _generate(tmp_path, tracker=TrackerChoice.WANDB, wandb_entity="my-team")
    wandb_config = (proj / proj.name / "configs" / "tracker" / "wandb.yaml").read_text()
    assert "entity: my-team" in wandb_config
    assert "__WANDB_ENTITY__" not in wandb_config
