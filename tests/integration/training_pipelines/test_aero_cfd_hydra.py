#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.integration.training_pipelines.fixtures.helpers import run_hydra_recipe

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AERO_CFD_CONFIGS = str(_REPO_ROOT / "recipes" / "aero_cfd" / "configs")

_STUB_SHAPENET = "tests.integration.training_pipelines.fixtures.synthetic_datasets.StubShapeNetCarDataset"
_STUB_DRIVAERML = "tests.integration.training_pipelines.fixtures.synthetic_datasets.StubDrivAerMLDataset"


@pytest.mark.usefixtures("aero_cfd_on_path")
@pytest.mark.parametrize("model_name", ["transformer", "transolver", "ab_upt", "upt"])
def test_shapenet_pipeline(model_name: str, tmp_path: Path, accelerator: str, device: str) -> None:
    """ShapeNet-Car recipe runs end-to-end and updates weights for each model architecture."""
    run_hydra_recipe(
        configs_dir=_AERO_CFD_CONFIGS,
        config_name="train_shapenet",
        overrides=[f"+experiment/shapenet={model_name}"],
        stub_kind=_STUB_SHAPENET,
        tmp_path=tmp_path,
        accelerator=accelerator,
        device=device,
        label=f"shapenet/{model_name}",
    )


@pytest.mark.usefixtures("aero_cfd_on_path")
@pytest.mark.parametrize("model_name", ["transformer", "transolver", "ab_upt", "upt"])
def test_drivaerml_pipeline(model_name: str, tmp_path: Path, accelerator: str, device: str) -> None:
    """DrivAerML recipe runs end-to-end and updates weights for each model architecture."""
    run_hydra_recipe(
        configs_dir=_AERO_CFD_CONFIGS,
        config_name="train_drivaerml",
        overrides=[f"+experiment/drivaerml={model_name}"],
        stub_kind=_STUB_DRIVAERML,
        tmp_path=tmp_path,
        accelerator=accelerator,
        device=device,
        label=f"drivaerml/{model_name}",
    )


# Orthogonal cases — exercised on the cheapest combo (shapenet × transformer)
# rather than the full grid, since they're testing trainer mechanics rather
# than recipe wiring.


@pytest.mark.usefixtures("aero_cfd_on_path")
def test_shapenet_with_gradient_accumulation(tmp_path: Path, accelerator: str, device: str) -> None:
    """Effective batch size > max batch size triggers gradient accumulation."""
    run_hydra_recipe(
        configs_dir=_AERO_CFD_CONFIGS,
        config_name="train_shapenet",
        overrides=["+experiment/shapenet=transformer"],
        stub_kind=_STUB_SHAPENET,
        tmp_path=tmp_path,
        accelerator=accelerator,
        device=device,
        label="shapenet/transformer+grad_accum",
        extra={
            "trainer.effective_batch_size": 2,
            "trainer.max_batch_size": 1,
            "trainer.disable_gradient_accumulation": False,
        },
    )


@pytest.mark.gpu
@pytest.mark.usefixtures("aero_cfd_on_path")
def test_shapenet_with_bf16(tmp_path: Path) -> None:
    """Mixed-precision (bfloat16) training runs end-to-end. GPU only."""
    if not torch.cuda.is_available():
        pytest.skip("bfloat16 mixed-precision training requires a CUDA GPU")
    run_hydra_recipe(
        configs_dir=_AERO_CFD_CONFIGS,
        config_name="train_shapenet",
        overrides=["+experiment/shapenet=transformer"],
        stub_kind=_STUB_SHAPENET,
        tmp_path=tmp_path,
        accelerator="gpu",
        device="cuda",
        label="shapenet/transformer+bf16",
        extra={"trainer.precision": "bfloat16"},
    )


@pytest.mark.usefixtures("aero_cfd_on_path")
def test_shapenet_with_default_callbacks(tmp_path: Path, accelerator: str, device: str) -> None:
    """Trainer wiring still works when default + trainer callbacks are enabled."""
    # Re-enable the default callbacks the override helper turns off. Keep the
    # user callback list empty so we don't pull in the production
    # OfflineLossCallback that needs a 'test' dataset we've dropped.
    run_hydra_recipe(
        configs_dir=_AERO_CFD_CONFIGS,
        config_name="train_shapenet",
        overrides=["+experiment/shapenet=transformer"],
        stub_kind=_STUB_SHAPENET,
        tmp_path=tmp_path,
        accelerator=accelerator,
        device=device,
        label="shapenet/transformer+default_callbacks",
        extra={
            "trainer.add_default_callbacks": True,
            "trainer.add_trainer_callbacks": True,
        },
    )
