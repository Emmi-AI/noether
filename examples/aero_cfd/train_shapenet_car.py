#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from examples.aero_cfd import ShapeNetCarPreset
from noether.training.runners import HydraRunner

DATASET_ROOT = "/path/to/shapenet_car"
OUTPUT_PATH = "/path/to/outputs/shapenet_car"
TRAINER_KIND = "noether.training.trainers.WeightedLossTrainer"
FIELD_WEIGHTS = {"surface_pressure": 1.0, "volume_velocity": 1.0}


def train_abupt(
    *,
    dataset_root: str = DATASET_ROOT,
    output_path: str = OUTPUT_PATH,
    accelerator: str = "mps",
    device: str = "mps",
) -> None:
    """Trains AB-UPT model using ShapeNetCar dataset."""
    preset = ShapeNetCarPreset()
    config = preset.build_config(
        model_kind="noether.modeling.models.aerodynamics.AeroABUPT",
        model_params=dict(hidden_dim=192, geometry_depth=6, physics_blocks=["perceiver"] + ["shared", "cross"] * 5),
        trainer_kind=TRAINER_KIND,
        trainer_params=dict(field_weights=FIELD_WEIGHTS),
        dataset_root=dataset_root,
        output_path=output_path,
        max_epochs=2,
        accelerator=accelerator,
    )
    HydraRunner().main(device=device, config=config)


def train_upt(
    *,
    dataset_root: str = DATASET_ROOT,
    output_path: str = OUTPUT_PATH,
    accelerator: str = "mps",
    device: str = "mps",
) -> None:
    """Trains UPT model using ShapeNetCar dataset."""
    preset = ShapeNetCarPreset()
    config = preset.build_config(
        model_kind="noether.modeling.models.aerodynamics.AeroUPT",
        model_params=dict(hidden_dim=192, num_heads=3, approximator_depth=12),
        trainer_kind=TRAINER_KIND,
        trainer_params=dict(field_weights=FIELD_WEIGHTS),
        dataset_root=dataset_root,
        output_path=output_path,
        max_epochs=2,
        accelerator=accelerator,
    )
    HydraRunner().main(device=device, config=config)


def train_transformer(
    *,
    dataset_root: str = DATASET_ROOT,
    output_path: str = OUTPUT_PATH,
    accelerator: str = "mps",
    device: str = "mps",
) -> None:
    """Trains Transformer model using ShapeNetCar dataset."""
    preset = ShapeNetCarPreset()
    config = preset.build_config(
        model_kind="noether.modeling.models.aerodynamics.AeroTransformer",
        model_params=dict(hidden_dim=192, depth=12),
        trainer_kind=TRAINER_KIND,
        trainer_params=dict(field_weights=FIELD_WEIGHTS),
        dataset_root=dataset_root,
        output_path=output_path,
        max_epochs=2,
        accelerator=accelerator,
    )
    HydraRunner().main(device=device, config=config)


def train_transolver(
    *,
    dataset_root: str = DATASET_ROOT,
    output_path: str = OUTPUT_PATH,
    accelerator: str = "mps",
    device: str = "mps",
) -> None:
    """Trains Transolver model using ShapeNetCar dataset."""
    preset = ShapeNetCarPreset()
    config = preset.build_config(
        model_kind="noether.modeling.models.aerodynamics.AeroTransolver",
        model_params=dict(hidden_dim=192, depth=12, num_slices=512),
        trainer_kind=TRAINER_KIND,
        trainer_params=dict(field_weights=FIELD_WEIGHTS),
        dataset_root=dataset_root,
        output_path=output_path,
        max_epochs=2,
        accelerator=accelerator,
    )
    HydraRunner().main(device=device, config=config)


if __name__ == "__main__":
    train_abupt()
    # train_upt()
    # train_transformer()
    # train_transolver()
