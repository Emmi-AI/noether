#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from pathlib import Path

from aero_cfd.presets import EmmiWingPreset

from noether.core.distributed.utils import accelerator_to_device
from noether.data.datasets.cfd.emmi_wing.dataset_hf import EmmiWingHFDataset
from noether.training.runners import HydraRunner

TRAINER_KIND = "noether.training.trainers.WeightedLossTrainer"
FIELD_WEIGHTS = {
    "surface_pressure": 1.0,
    "surface_friction": 1.0,
    "volume_pressure": 1.0,
    "volume_velocity": 1.0,
    "volume_vorticity": 1.0,
}

DEFAULT_HF_CACHE = "~/.cache/noether/emmi_wing_hf"


def _get_preset(use_hf: bool):  # noqa: ANN202
    preset = EmmiWingPreset()
    if use_hf:
        preset.dataset_kind = "noether.data.datasets.cfd.EmmiWingHFDataset"
    return preset


def _ensure_dataset(dataset_root: str | None) -> tuple[str, bool]:
    """Return (dataset_root, use_hf).  Downloads from HF if needed.

    If ``dataset_root`` is provided and contains data, uses it as-is with the
    full dataset splits.  If the directory is empty or missing, downloads the
    HF subset into it.  If ``dataset_root`` is None, uses the default cache.
    """
    local_dir = str(Path(dataset_root or DEFAULT_HF_CACHE).expanduser())
    local_path = Path(local_dir)

    # If the directory already has run_* data, assume it's ready:
    if local_path.exists() and list(local_path.glob("run_*")):
        # Use full dataset splits if this looks like the full dataset (>248 runs):
        use_hf = len(list(local_path.glob("run_*"))) <= 248
        return local_dir, use_hf

    # Download HF subset:
    EmmiWingHFDataset.download(local_dir)
    return local_dir, True


def train_abupt(
    *,
    dataset_root: str,
    output_path: str,
    accelerator: str = "gpu",
    precision: str = "float32",
    max_epochs: int = 100,
    use_hf: bool = False,
) -> None:
    """Train AB-UPT on Emmi-Wing."""
    preset = _get_preset(use_hf)
    config = preset.build_config(
        model_kind="noether.modeling.models.aerodynamics.AeroABUPT",
        model_params=dict(hidden_dim=192, geometry_depth=6, physics_blocks=["perceiver"] + ["shared", "cross"] * 5),
        trainer_kind=TRAINER_KIND,
        trainer_params=dict(field_weights=FIELD_WEIGHTS, precision=precision),
        dataset_root=dataset_root,
        output_path=output_path,
        datasets=["train", "val", "test"],
        max_epochs=max_epochs,
        accelerator=accelerator,
    )
    HydraRunner().main(device=accelerator_to_device(accelerator), config=config)


def train_upt(
    *,
    dataset_root: str,
    output_path: str,
    accelerator: str = "gpu",
    precision: str = "float32",
    max_epochs: int = 100,
    use_hf: bool = False,
) -> None:
    """Train UPT on Emmi-Wing."""
    preset = _get_preset(use_hf)
    config = preset.build_config(
        model_kind="noether.modeling.models.aerodynamics.AeroUPT",
        model_params=dict(hidden_dim=192, num_heads=3, approximator_depth=12),
        trainer_kind=TRAINER_KIND,
        trainer_params=dict(field_weights=FIELD_WEIGHTS, precision=precision),
        dataset_root=dataset_root,
        output_path=output_path,
        datasets=["train", "val", "test"],
        max_epochs=max_epochs,
        accelerator=accelerator,
    )
    HydraRunner().main(device=accelerator_to_device(accelerator), config=config)


def train_transformer(
    *,
    dataset_root: str,
    output_path: str,
    accelerator: str = "gpu",
    precision: str = "float32",
    max_epochs: int = 100,
    use_hf: bool = False,
) -> None:
    """Train Transformer on Emmi-Wing."""
    preset = _get_preset(use_hf)
    config = preset.build_config(
        model_kind="noether.modeling.models.aerodynamics.AeroTransformer",
        model_params=dict(hidden_dim=192, depth=12),
        trainer_kind=TRAINER_KIND,
        trainer_params=dict(field_weights=FIELD_WEIGHTS, precision=precision),
        dataset_root=dataset_root,
        output_path=output_path,
        datasets=["train", "val", "test"],
        max_epochs=max_epochs,
        accelerator=accelerator,
    )
    HydraRunner().main(device=accelerator_to_device(accelerator), config=config)


def train_transolver(
    *,
    dataset_root: str,
    output_path: str,
    accelerator: str = "gpu",
    precision: str = "float32",
    max_epochs: int = 100,
    use_hf: bool = False,
) -> None:
    """Train Transolver on Emmi-Wing."""
    preset = _get_preset(use_hf)
    config = preset.build_config(
        model_kind="noether.modeling.models.aerodynamics.AeroTransolver",
        model_params=dict(hidden_dim=192, depth=12, num_slices=512),
        trainer_kind=TRAINER_KIND,
        trainer_params=dict(field_weights=FIELD_WEIGHTS, precision=precision),
        dataset_root=dataset_root,
        output_path=output_path,
        datasets=["train", "val", "test"],
        max_epochs=max_epochs,
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

    parser = argparse.ArgumentParser(
        description="Train aerodynamic models on Emmi-Wing dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # Auto-download HF subset and train with defaults:\n"
            "  python scripts/train_emmi_wing.py --output-path ./outputs\n"
            "\n"
            "  # AB-UPT on GPU with mixed precision:\n"
            "  python scripts/train_emmi_wing.py --model abupt --accelerator gpu --precision float16 --max-epochs 500 --output-path ./outputs\n"
            "\n"
            "  # AB-UPT on Apple Silicon:\n"
            "  python scripts/train_emmi_wing.py --model abupt --accelerator mps --max-epochs 100 --output-path ./outputs\n"
            "\n"
            "  # Full dataset (local copy):\n"
            "  python scripts/train_emmi_wing.py --dataset-root /path/to/emmi_wings --model abupt --accelerator gpu --precision float16 --max-epochs 500 --output-path ./outputs\n"
            "\n"
            "  # Download HF subset to a custom directory:\n"
            "  python scripts/train_emmi_wing.py --dataset-root /my/data/emmi_wing_hf --model abupt --accelerator gpu --max-epochs 100 --output-path ./outputs\n"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Path to local Emmi-Wing dataset. If omitted, downloads from HuggingFace.",
    )
    parser.add_argument("--output-path", default="./outputs", help="Path to store training outputs.")
    parser.add_argument("--accelerator", default="gpu", choices=["cpu", "gpu", "mps"], help="Accelerator to use.")
    parser.add_argument("--model", default="abupt", choices=list(MODELS), help="Model architecture to train.")
    parser.add_argument(
        "--precision", default="float32", choices=["float32", "float16", "bfloat16"], help="Training precision."
    )
    parser.add_argument("--max-epochs", type=int, default=100, help="Maximum training epochs.")
    args = parser.parse_args()

    dataset_root, use_hf = _ensure_dataset(args.dataset_root)

    MODELS[args.model](
        dataset_root=dataset_root,
        output_path=args.output_path,
        accelerator=args.accelerator,
        precision=args.precision,
        max_epochs=args.max_epochs,
        use_hf=use_hf,
    )
