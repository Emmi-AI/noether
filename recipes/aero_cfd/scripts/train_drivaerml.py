#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from aero_cfd.presets import DrivAerMLPreset

from noether.core.distributed.utils import accelerator_to_device
from noether.training.runners import HydraRunner

TRAINER_KIND = "noether.training.trainers.WeightedLossTrainer"
FIELD_WEIGHTS = {
    "surface_pressure": 1.0,
    "surface_friction": 1.0,
    "volume_pressure": 1.0,
    "volume_velocity": 1.0,
    "volume_vorticity": 1.0,
}


def train_abupt(
    *,
    dataset_root: str,
    output_path: str,
    accelerator: str = "gpu",
) -> None:
    """Train AB-UPT model using DrivAerML dataset."""
    preset = DrivAerMLPreset()
    config = preset.build_config(
        model_kind="noether.modeling.models.aerodynamics.AeroABUPT",
        model_params=dict(hidden_dim=192, geometry_depth=6, physics_blocks=["perceiver"] + ["shared", "cross"] * 5),
        trainer_kind=TRAINER_KIND,
        trainer_params=dict(field_weights=FIELD_WEIGHTS),
        dataset_root=dataset_root,
        output_path=output_path,
        datasets=["train", "val", "test"],
        max_epochs=2,
        accelerator=accelerator,
        num_workers=0,
    )
    HydraRunner().main(device=accelerator_to_device(accelerator), config=config)


def train_upt(
    *,
    dataset_root: str,
    output_path: str,
    accelerator: str = "gpu",
) -> None:
    """Train UPT model using DrivAerML dataset."""
    preset = DrivAerMLPreset()
    config = preset.build_config(
        model_kind="noether.modeling.models.aerodynamics.AeroUPT",
        model_params=dict(hidden_dim=192, num_heads=3, approximator_depth=12),
        trainer_kind=TRAINER_KIND,
        trainer_params=dict(field_weights=FIELD_WEIGHTS),
        dataset_root=dataset_root,
        output_path=output_path,
        datasets=["train", "val", "test"],
        max_epochs=2,
        accelerator=accelerator,
    )
    HydraRunner().main(device=accelerator_to_device(accelerator), config=config)


def train_transformer(
    *,
    dataset_root: str,
    output_path: str,
    accelerator: str = "gpu",
) -> None:
    """Train Transformer model using DrivAerML dataset."""
    preset = DrivAerMLPreset()
    config = preset.build_config(
        model_kind="noether.modeling.models.aerodynamics.AeroTransformer",
        model_params=dict(hidden_dim=192, depth=12),
        trainer_kind=TRAINER_KIND,
        trainer_params=dict(field_weights=FIELD_WEIGHTS),
        dataset_root=dataset_root,
        output_path=output_path,
        datasets=["train", "val", "test"],
        max_epochs=2,
        accelerator=accelerator,
    )
    HydraRunner().main(device=accelerator_to_device(accelerator), config=config)


def train_transolver(
    *,
    dataset_root: str,
    output_path: str,
    accelerator: str = "gpu",
) -> None:
    """Train Transolver model using DrivAerML dataset."""
    preset = DrivAerMLPreset()
    config = preset.build_config(
        model_kind="noether.modeling.models.aerodynamics.AeroTransolver",
        model_params=dict(hidden_dim=192, depth=12, num_slices=512),
        trainer_kind=TRAINER_KIND,
        trainer_params=dict(field_weights=FIELD_WEIGHTS),
        dataset_root=dataset_root,
        output_path=output_path,
        datasets=["train", "val", "test"],
        max_epochs=2,
        accelerator=accelerator,
    )
    HydraRunner().main(device=accelerator_to_device(accelerator), config=config)


MODELS = {
    "abupt": train_abupt,
    "upt": train_upt,
    "transformer": train_transformer,
    "transolver": train_transolver,
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train aerodynamic models on DrivAerML dataset.")
    parser.add_argument("--dataset-root", required=True, help="Path to the DrivAerML dataset.")
    parser.add_argument("--output-path", required=True, help="Path to store training outputs.")
    parser.add_argument("--accelerator", default="gpu", choices=["cpu", "gpu", "mps"], help="Accelerator to use.")
    parser.add_argument("--model", default="abupt", choices=list(MODELS), help="Model architecture to train.")
    args = parser.parse_args()

    MODELS[args.model](dataset_root=args.dataset_root, output_path=args.output_path, accelerator=args.accelerator)
