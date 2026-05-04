#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from noether.core.schemas.schema import ConfigSchema
from tests.integration.training_pipelines.fixtures.helpers import (
    make_run_dirs,
    train_and_assert_weights_changed,
)
from tests.integration.training_pipelines.fixtures.synthetic_datasets import StubShapeNetCarDataset

_TRAINER_KIND = "noether.training.trainers.WeightedLossTrainer"

_SHAPENET_FIELD_WEIGHTS = {"surface_pressure": 1.0, "volume_velocity": 1.0}
_DRIVAERML_FIELD_WEIGHTS = {
    "surface_pressure": 1.0,
    "surface_friction": 1.0,
    "volume_pressure": 1.0,
    "volume_velocity": 1.0,
    "volume_vorticity": 1.0,
}


@pytest.fixture
def stubbed_shapenet_preset(aero_cfd_on_path):
    """Yield a ``ShapeNetCarPreset`` subclass whose ``dataset_kind`` is the stub."""
    from aero_cfd.presets import ShapeNetCarPreset  # imported under aero_cfd_on_path

    class _StubbedShapeNetCarPreset(ShapeNetCarPreset):
        dataset_kind = "tests.integration.training_pipelines.fixtures.synthetic_datasets.StubShapeNetCarDataset"

    return _StubbedShapeNetCarPreset()


@pytest.fixture
def stubbed_drivaerml_preset(aero_cfd_on_path):
    """Yield a ``DrivAerMLPreset`` subclass whose ``dataset_kind`` is the stub."""
    from aero_cfd.presets import DrivAerMLPreset

    class _StubbedDrivAerMLPreset(DrivAerMLPreset):
        dataset_kind = "tests.integration.training_pipelines.fixtures.synthetic_datasets.StubDrivAerMLDataset"

    return _StubbedDrivAerMLPreset()


def _patch_for_test(config: ConfigSchema, *, extra_excluded_properties: set[str] | None = None) -> None:
    """Strip the parts of a preset-built config that don't belong in an in-process test.

    The preset's ``build_config`` falls through to ``standard_callbacks(...)``
    on an empty ``callbacks_override`` (Python ``[] or default`` quirk), so the
    only reliable way to disable callbacks is to overwrite the trainer config
    after the fact.
    """
    config.trainer.callbacks = []
    config.trainer.add_default_callbacks = False
    config.trainer.add_trainer_callbacks = False
    config.num_workers = 0
    config.store_code_in_output = False
    config.tracker = None
    config.slurm = None
    if extra_excluded_properties:
        train_ds_cfg = config.datasets["train"]
        train_ds_cfg.excluded_properties = (train_ds_cfg.excluded_properties or set()) | extra_excluded_properties


def _build_kwargs(
    *,
    model_kind: str,
    field_weights: dict[str, float],
    dataset_root: Path,
    output_path: Path,
    accelerator: str,
) -> dict[str, Any]:
    return dict(
        model_kind=model_kind,
        model_params=dict(hidden_dim=96, depth=2),
        trainer_kind=_TRAINER_KIND,
        trainer_params=dict(field_weights=field_weights),
        dataset_root=str(dataset_root),
        output_path=str(output_path),
        accelerator=accelerator,
        max_epochs=1,
        batch_size=1,
        datasets=["train"],
    )


@pytest.mark.parametrize(
    "model_kind",
    [
        "noether.modeling.models.aerodynamics.AeroTransformer",
        "noether.modeling.models.aerodynamics.AeroTransolver",
    ],
)
def test_shapenet_python_api_pipeline(
    model_kind: str,
    stubbed_shapenet_preset,
    tmp_path: Path,
    accelerator: str,
    device: str,
) -> None:
    """End-to-end ShapeNetCar run via the preset API."""
    output_path, dataset_root = make_run_dirs(tmp_path)

    config = stubbed_shapenet_preset.build_config(
        include_evaluation=False,
        **_build_kwargs(
            model_kind=model_kind,
            field_weights=_SHAPENET_FIELD_WEIGHTS,
            dataset_root=dataset_root,
            output_path=output_path,
            accelerator=accelerator,
        ),
    )

    assert config.datasets["train"].kind.endswith("StubShapeNetCarDataset")

    # ShapeNetCarPreset.excluded_properties omits ``surface_area``, but the
    # SHAPENET_CAR_FILEMAP doesn't define a ``surface_area`` file either — so
    # ``getitem_surface_area`` calls ``_load(idx, None)`` and the production
    # code would also fail (it just isn't reached today because the YAML config
    # path adds it to excluded_properties). Exclude it here so the test doesn't
    # trip over a recipe inconsistency unrelated to the pipeline.
    _patch_for_test(config, extra_excluded_properties={"surface_area"})
    train_and_assert_weights_changed(config, device=device, label=f"python-api shapenet/{model_kind}")


@pytest.mark.parametrize(
    "model_kind",
    [
        "noether.modeling.models.aerodynamics.AeroTransformer",
        "noether.modeling.models.aerodynamics.AeroTransolver",
    ],
)
def test_drivaerml_python_api_pipeline(
    model_kind: str,
    stubbed_drivaerml_preset,
    tmp_path: Path,
    accelerator: str,
    device: str,
) -> None:
    """End-to-end DrivAerML run via the preset API."""
    output_path, dataset_root = make_run_dirs(tmp_path)

    config = stubbed_drivaerml_preset.build_config(
        **_build_kwargs(
            model_kind=model_kind,
            field_weights=_DRIVAERML_FIELD_WEIGHTS,
            dataset_root=dataset_root,
            output_path=output_path,
            accelerator=accelerator,
        ),
    )

    assert config.datasets["train"].kind.endswith("StubDrivAerMLDataset")

    _patch_for_test(config)
    train_and_assert_weights_changed(config, device=device, label=f"python-api drivaerml/{model_kind}")


def test_stub_dataset_class_path_is_importable() -> None:
    """The dotted path embedded in the stubbed preset must remain valid."""
    from noether.core.factory import class_constructor_from_class_path

    cls = class_constructor_from_class_path(
        "tests.integration.training_pipelines.fixtures.synthetic_datasets.StubShapeNetCarDataset"
    )
    assert cls is StubShapeNetCarDataset
