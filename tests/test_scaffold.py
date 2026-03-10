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
# Generate a representative subset: each model with a shapenet and a caeml dataset
COMBOS = list(itertools.product(MODELS, [DatasetChoice.SHAPENET_CAR, DatasetChoice.AHMEDML]))


@pytest.mark.parametrize(("model", "dataset"), COMBOS, ids=[f"{m.value}-{d.value}" for m, d in COMBOS])
def test_generate_project(tmp_path: Path, model: ModelChoice, dataset: DatasetChoice) -> None:
    project_name = "test_proj"
    proj = tmp_path / project_name
    config = resolve_config(
        project_name=project_name,
        model=model,
        dataset=dataset,
        dataset_path="/tmp/fake_data",
        optimizer=OptimizerChoice.ADAMW,
        tracker=TrackerChoice.DISABLED,
        hardware=HardwareChoice.GPU,
        project_dir=proj,
        wandb_entity=None,
    )

    generate_project(config)

    # All expected directories exist
    assert (proj / "configs").is_dir()
    assert (proj / "model").is_dir()
    assert (proj / "schemas").is_dir()

    # No leftover tutorial references in generated .py or .yaml files
    for ext in ("*.py", "*.yaml"):
        for f in proj.rglob(ext):
            content = f.read_text()
            assert "tutorial." not in content, f"Found 'tutorial.' in {f.relative_to(proj)}"

    # All YAML files parse without error
    for yf in proj.rglob("*.yaml"):
        content = yf.read_text()
        if not content.strip():
            continue
        # Strip Hydra directives before parsing
        lines = [line for line in content.splitlines() if not line.startswith("# @package")]
        try:
            yaml.safe_load("\n".join(lines))
        except yaml.YAMLError as e:
            pytest.fail(f"YAML parse error in {yf.relative_to(proj)}: {e}")

    # All kind: values start with project name, a known framework prefix, or are Hydra interpolations
    known_prefixes = (f"{project_name}.", "noether.", "torch.", "${")
    for yf in proj.rglob("*.yaml"):
        content = yf.read_text()
        lines = content.splitlines()
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("kind:"):
                kind_value = stripped.split(":", 1)[1].strip().strip("'\"")
                if kind_value:
                    assert any(kind_value.startswith(p) for p in known_prefixes), (
                        f"Unexpected kind '{kind_value}' in {yf.relative_to(proj)}"
                    )

    # No unresolved placeholders
    for ext in ("*.py", "*.yaml"):
        for f in proj.rglob(ext):
            content = f.read_text()
            assert "__PROJECT__" not in content, f"Unresolved __PROJECT__ in {f.relative_to(proj)}"
            assert "__CLASS__" not in content, f"Unresolved __CLASS__ in {f.relative_to(proj)}"
            assert "__DATASET_PATH__" not in content, f"Unresolved __DATASET_PATH__ in {f.relative_to(proj)}"
            assert "__OPTIMIZER__" not in content, f"Unresolved __OPTIMIZER__ in {f.relative_to(proj)}"
            assert "__TRACKER__" not in content, f"Unresolved __TRACKER__ in {f.relative_to(proj)}"


def test_hardware_mps_sets_accelerator(tmp_path: Path) -> None:
    """Non-default hardware should write accelerator to train.yaml."""
    config = resolve_config(
        project_name="mps_test",
        model=ModelChoice.UPT,
        dataset=DatasetChoice.SHAPENET_CAR,
        dataset_path="/tmp/fake",
        optimizer=OptimizerChoice.ADAMW,
        tracker=TrackerChoice.DISABLED,
        hardware=HardwareChoice.MPS,
        project_dir=tmp_path / "mps_test",
        wandb_entity=None,
    )

    generate_project(config)

    train_yaml = tmp_path / "mps_test" / "configs" / "train.yaml"
    lines = [line for line in train_yaml.read_text().splitlines() if not line.startswith("# @package")]
    data = yaml.safe_load("\n".join(lines))
    assert data.get("accelerator") == "mps"


def test_gpu_default_no_accelerator(tmp_path: Path) -> None:
    """Default GPU hardware should not write accelerator key."""
    config = resolve_config(
        project_name="gpu_test",
        model=ModelChoice.UPT,
        dataset=DatasetChoice.SHAPENET_CAR,
        dataset_path="/tmp/fake",
        optimizer=OptimizerChoice.ADAMW,
        tracker=TrackerChoice.DISABLED,
        hardware=HardwareChoice.GPU,
        project_dir=tmp_path / "gpu_test",
        wandb_entity=None,
    )

    generate_project(config)

    train_yaml = tmp_path / "gpu_test" / "configs" / "train.yaml"
    lines = [line for line in train_yaml.read_text().splitlines() if not line.startswith("# @package")]
    data = yaml.safe_load("\n".join(lines))
    assert "accelerator" not in data
