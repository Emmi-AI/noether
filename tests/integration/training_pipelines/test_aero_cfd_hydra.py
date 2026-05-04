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
_AERO_CFD_CONFIGS = str(_REPO_ROOT / "recipes" / "aero_cfd" / "configs")

_STUB_SHAPENET = "tests.integration.training_pipelines.fixtures.synthetic_datasets.StubShapeNetCarDataset"
_STUB_DRIVAERML = "tests.integration.training_pipelines.fixtures.synthetic_datasets.StubDrivAerMLDataset"


def _instantiate_schema(runner_dict: dict) -> ConfigSchema:
    schema_kind = runner_dict.get("config_schema_kind")
    schema_cls: type[ConfigSchema] = class_constructor_from_class_path(schema_kind) if schema_kind else ConfigSchema
    return schema_cls(**runner_dict)


def _floating_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    # Snapshot on CPU: setup_experiment returns a CPU model, but trainer.train
    # moves it to self.device, so post-train state lives on the accelerator.
    # Comparing across devices via torch.equal raises; staying on CPU keeps
    # the assertion device-agnostic.
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if v.is_floating_point()}


def _any_changed(before: dict[str, torch.Tensor], after_state: dict) -> bool:
    return any(not torch.equal(before[k], after_state[k].detach().cpu()) for k in before)


def _run_recipe(
    *,
    config_name: str,
    experiment_override: str,
    stub_kind: str,
    tmp_path: Path,
    accelerator: str,
    device: str,
    extra: dict | None = None,
) -> tuple[torch.nn.Module, bool]:
    output_path = tmp_path / "out"
    dataset_root = tmp_path / "data"
    output_path.mkdir()
    dataset_root.mkdir()

    with initialize_config_dir(version_base=None, config_dir=_AERO_CFD_CONFIGS, job_name="test"):
        cfg = compose(
            config_name=config_name,
            overrides=[experiment_override, "tracker=disabled"],
        )

    apply_test_overrides(
        cfg,
        accelerator=accelerator,
        output_path=output_path,
        dataset_root=dataset_root,
        stub_dataset_kind=stub_kind,
        extra=extra,
    )

    config_obj = _instantiate_schema(to_runner_dict(cfg))
    trainer, model, _tracker, _mc = HydraRunner.setup_experiment(device=device, config=config_obj)

    before = _floating_state(model)
    trainer.train(model)
    changed = _any_changed(before, model.state_dict())
    return model, changed


@pytest.mark.usefixtures("aero_cfd_on_path")
@pytest.mark.parametrize("model_name", ["transformer", "transolver", "ab_upt", "upt"])
def test_shapenet_pipeline(model_name: str, tmp_path: Path, accelerator: str, device: str) -> None:
    """ShapeNet-Car recipe runs end-to-end and updates weights for each model architecture."""
    _, changed = _run_recipe(
        config_name="train_shapenet",
        experiment_override=f"+experiment/shapenet={model_name}",
        stub_kind=_STUB_SHAPENET,
        tmp_path=tmp_path,
        accelerator=accelerator,
        device=device,
    )
    assert changed, f"no parameter changed during shapenet/{model_name} training"


@pytest.mark.usefixtures("aero_cfd_on_path")
@pytest.mark.parametrize("model_name", ["transformer", "transolver", "ab_upt", "upt"])
def test_drivaerml_pipeline(model_name: str, tmp_path: Path, accelerator: str, device: str) -> None:
    """DrivAerML recipe runs end-to-end and updates weights for each model architecture."""
    _, changed = _run_recipe(
        config_name="train_drivaerml",
        experiment_override=f"+experiment/drivaerml={model_name}",
        stub_kind=_STUB_DRIVAERML,
        tmp_path=tmp_path,
        accelerator=accelerator,
        device=device,
    )
    assert changed, f"no parameter changed during drivaerml/{model_name} training"


# Orthogonal cases — exercised on the cheapest combo (shapenet × transformer)
# rather than the full grid, since they're testing trainer mechanics rather
# than recipe wiring.


@pytest.mark.usefixtures("aero_cfd_on_path")
def test_shapenet_with_gradient_accumulation(tmp_path: Path, accelerator: str, device: str) -> None:
    """Effective batch size > max batch size triggers gradient accumulation."""
    _, changed = _run_recipe(
        config_name="train_shapenet",
        experiment_override="+experiment/shapenet=transformer",
        stub_kind=_STUB_SHAPENET,
        tmp_path=tmp_path,
        accelerator=accelerator,
        device=device,
        extra={
            "trainer.effective_batch_size": 2,
            "trainer.max_batch_size": 1,
            "trainer.disable_gradient_accumulation": False,
        },
    )
    assert changed, "no parameter changed when running with gradient accumulation"


@pytest.mark.gpu
def test_shapenet_with_bf16(tmp_path: Path) -> None:
    """Mixed-precision (bfloat16) training runs end-to-end. GPU only."""
    if not torch.cuda.is_available():
        pytest.skip("bfloat16 mixed-precision training requires a CUDA GPU")
    _, changed = _run_recipe(
        config_name="train_shapenet",
        experiment_override="+experiment/shapenet=transformer",
        stub_kind=_STUB_SHAPENET,
        tmp_path=tmp_path,
        accelerator="gpu",
        device="cuda",
        extra={"trainer.precision": "bfloat16"},
    )
    assert changed, "no parameter changed when running with bfloat16 precision"


@pytest.mark.usefixtures("aero_cfd_on_path")
def test_shapenet_with_default_callbacks(tmp_path: Path, accelerator: str, device: str) -> None:
    """Trainer wiring still works when default + trainer callbacks are enabled."""
    # Re-enable the default callbacks the override helper turns off. Keep the
    # user callback list empty so we don't pull in the production
    # OfflineLossCallback that needs a 'test' dataset we've dropped.
    _, changed = _run_recipe(
        config_name="train_shapenet",
        experiment_override="+experiment/shapenet=transformer",
        stub_kind=_STUB_SHAPENET,
        tmp_path=tmp_path,
        accelerator=accelerator,
        device=device,
        extra={
            "trainer.add_default_callbacks": True,
            "trainer.add_trainer_callbacks": True,
        },
    )
    assert changed, "no parameter changed with default callbacks enabled"
