#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from noether.core.factory import class_constructor_from_class_path
from noether.core.schemas.schema import ConfigSchema
from noether.training.runners import HydraRunner
from tests.integration.training_pipelines.fixtures.overrides import apply_test_overrides, to_runner_dict

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HEAT_TRANSFER_CONFIGS = str(_REPO_ROOT / "recipes" / "heat_transfer" / "configs")
_STUB_HEATSINK = "tests.integration.training_pipelines.fixtures.synthetic_datasets.StubSimshiftHeatsinkDataset"


def _instantiate_schema(runner_dict: dict) -> ConfigSchema:
    schema_kind = runner_dict.get("config_schema_kind")
    schema_cls: type[ConfigSchema] = class_constructor_from_class_path(schema_kind) if schema_kind else ConfigSchema
    return schema_cls(**runner_dict)


def _floating_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items() if v.is_floating_point()}


def _any_changed(before: dict[str, torch.Tensor], after_state: dict) -> bool:
    return any(not torch.equal(before[k], after_state[k].detach()) for k in before)


@pytest.mark.usefixtures("heat_transfer_on_path")
@pytest.mark.parametrize("experiment_name", ["ab_upt", "transolver"])
def test_simshift_heatsink_pipeline(experiment_name: str, tmp_path: Path, accelerator: str, device: str) -> None:
    """SIMSHIFT-Heatsink recipe runs end-to-end and updates weights."""
    output_path = tmp_path / "out"
    dataset_root = tmp_path / "data"
    output_path.mkdir()
    dataset_root.mkdir()

    with initialize_config_dir(version_base=None, config_dir=_HEAT_TRANSFER_CONFIGS, job_name="test"):
        cfg = compose(
            config_name="train_simshift_heatsink",
            overrides=[
                f"+experiment/simshift_heatsink={experiment_name}",
                "tracker=disabled",
            ],
        )

    apply_test_overrides(
        cfg,
        accelerator=accelerator,
        output_path=output_path,
        dataset_root=dataset_root,
        stub_dataset_kind=_STUB_HEATSINK,
    )

    config_obj = _instantiate_schema(to_runner_dict(cfg))
    trainer, model, _tracker, _mc = HydraRunner.setup_experiment(device=device, config=config_obj)

    before = _floating_state(model)
    trainer.train(model)
    assert _any_changed(before, model.state_dict()), (
        f"no parameter changed during simshift_heatsink/{experiment_name} training"
    )
