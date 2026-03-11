#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from noether.scaffold.choices import DatasetChoice, HardwareChoice, ModelChoice, OptimizerChoice, TrackerChoice
from noether.scaffold.config import ScaffoldConfig, load_reference, resolve_config, substitute


def test_substitute_replaces_all_placeholders() -> None:
    config = ScaffoldConfig(
        project_name="my_proj",
        model=ModelChoice.UPT,
        dataset=DatasetChoice.SHAPENET_CAR,
        dataset_path="/data/shapenet",
        optimizer=OptimizerChoice.ADAMW,
        tracker=TrackerChoice.WANDB,
        hardware=HardwareChoice.GPU,
        project_dir=Path("/tmp/my_proj"),
        wandb_entity=None,
    )
    template = (
        "kind: __PROJECT__.model.UPT\ndataset_root: __DATASET_PATH__\noptimizer: __OPTIMIZER__\ntracker: __TRACKER__"
    )
    result = substitute(template, config)

    assert result == "kind: my_proj.model.UPT\ndataset_root: /data/shapenet\noptimizer: adamw\ntracker: wandb"


REFERENCE_KEYS = {
    "experiment_category",
    "data_specs_file",
    "normalizers_file",
    "statistics_file",
    "pipeline_file",
    "dataset_config_file",
    "trainer_config_file",
    "callbacks_file",
}


@pytest.mark.parametrize("dataset", list(DatasetChoice), ids=[d.value for d in DatasetChoice])
def test_load_reference_returns_expected_keys(dataset: DatasetChoice) -> None:
    ref = load_reference(dataset)
    assert isinstance(ref, dict)
    for key in REFERENCE_KEYS:
        assert key in ref, f"Missing key '{key}' in reference for {dataset.value}"


def test_resolve_config_populates_reference(tmp_path: Path) -> None:
    config = resolve_config(
        project_name="test_proj",
        model=ModelChoice.UPT,
        dataset=DatasetChoice.SHAPENET_CAR,
        dataset_path="/tmp/data",
        optimizer=OptimizerChoice.ADAMW,
        tracker=TrackerChoice.DISABLED,
        hardware=HardwareChoice.GPU,
        project_dir=tmp_path / "test_proj",
        wandb_entity=None,
    )
    assert config.project_name == "test_proj"
    assert config.model == ModelChoice.UPT
    assert config.dataset == DatasetChoice.SHAPENET_CAR
    assert config.dataset_path == "/tmp/data"
    assert config.optimizer == OptimizerChoice.ADAMW
    assert config.tracker == TrackerChoice.DISABLED
    assert config.hardware == HardwareChoice.GPU
    assert config.project_dir == tmp_path / "test_proj"
    assert config.wandb_entity is None
    assert REFERENCE_KEYS == config.reference.keys()
