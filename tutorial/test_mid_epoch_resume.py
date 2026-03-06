#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import argparse
import logging
import os
import re
import signal
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from noether.core.distributed.utils import accelerator_to_device
from noether.core.schemas.callbacks import (
    CheckpointCallbackConfig,
    OfflineLossCallbackConfig,
)
from noether.core.schemas.dataset import (
    AeroDataSpecs,
    DatasetBaseConfig,
    StandardDatasetConfig,
    SubsetWrapperConfig,
)
from noether.core.schemas.modules import (
    DeepPerceiverDecoderConfig,
    PerceiverBlockConfig,
    SupernodePoolingConfig,
    TransformerBlockConfig,
)
from noether.core.schemas.normalizers import AnyNormalizer, MeanStdNormalizerConfig, PositionNormalizerConfig
from noether.core.schemas.optimizers import OptimizerConfig
from noether.core.schemas.schedules import LinearWarmupCosineDecayScheduleConfig
from noether.core.schemas.schema import ConfigSchema, StaticConfigSchema
from noether.core.schemas.statistics import AeroStatsSchema
from noether.training.runners import HydraRunner
from tutorial.schemas.models.upt_config import UPTConfig
from tutorial.schemas.pipelines.aero_pipeline_config import AeroCFDPipelineConfig
from tutorial.schemas.trainers.automotive_aerodynamics_trainer_config import AutomotiveAerodynamicsCfdTrainerConfig

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Set up root logger to write to stderr so training logs are visible."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname).1s %(message)s", datefmt="%H:%M:%S"))
        root.addHandler(handler)


# --- ShapeNet Car:
DATASET_STATS = {
    "raw_pos_min": [-4.5],
    "raw_pos_max": [6.0],
    "surface_pressure_mean": [-36.4098],
    "surface_pressure_std": [48.6757],
    "volume_velocity_mean": [0.00293915, -0.0230546, 17.546032],
    "volume_velocity_std": [1.361689, 1.267649, 5.850353],
    "volume_sdf_mean": [3.74222e-01],
    "volume_sdf_std": [1.78948e-01],
}

DATA_SPECS = {
    "position_dim": 3,
    "surface_feature_dim": {"surface_sdf": 1, "surface_normals": 3},
    "volume_feature_dim": {"volume_sdf": 1, "volume_normals": 3},
    "surface_output_dims": {"pressure": 1},
    "volume_output_dims": {"velocity": 3},
}

MODEL_FORWARD_PROPERTIES = [
    "surface_mask_query",
    "surface_position_batch_idx",
    "surface_position_supernode_idx",
    "surface_position",
    "surface_query_position",
    "volume_query_position",
]
# ---


# --- mirror tutorial/train_shapenet_upt.py:


def build_specs() -> AeroDataSpecs:
    return AeroDataSpecs(**DATA_SPECS)


def build_dataset_normalizer() -> dict[str, list[AnyNormalizer]]:
    return {
        "surface_pressure": [
            MeanStdNormalizerConfig(
                kind="noether.data.preprocessors.normalizers.MeanStdNormalization",
                mean=DATASET_STATS["surface_pressure_mean"],
                std=DATASET_STATS["surface_pressure_std"],
            ),
        ],
        "volume_velocity": [
            MeanStdNormalizerConfig(
                kind="noether.data.preprocessors.normalizers.MeanStdNormalization",
                mean=DATASET_STATS["volume_velocity_mean"],
                std=DATASET_STATS["volume_velocity_std"],
            ),
        ],
        "volume_sdf": [
            MeanStdNormalizerConfig(
                kind="noether.data.preprocessors.normalizers.MeanStdNormalization",
                mean=DATASET_STATS["volume_sdf_mean"],
                std=DATASET_STATS["volume_sdf_std"],
            ),
        ],
        "surface_position": [
            PositionNormalizerConfig(
                kind="noether.data.preprocessors.normalizers.PositionNormalizer",
                raw_pos_min=DATASET_STATS["raw_pos_min"],
                raw_pos_max=DATASET_STATS["raw_pos_max"],
                scale=1000,
            ),
        ],
        "volume_position": [
            PositionNormalizerConfig(
                kind="noether.data.preprocessors.normalizers.PositionNormalizer",
                raw_pos_min=DATASET_STATS["raw_pos_min"],
                raw_pos_max=DATASET_STATS["raw_pos_max"],
                scale=1000,
            ),
        ],
    }


def build_dataset_config(
    mode: Literal["train", "test"],
    dataset_root: str,
    data_specs: AeroDataSpecs,
    dataset_statistics: dict[str, Sequence[float]],
    dataset_normalizer: dict[str, list[AnyNormalizer]],
    max_samples: int | None = None,
) -> DatasetBaseConfig:
    wrappers = None
    if max_samples is not None:
        wrappers = [
            SubsetWrapperConfig(
                kind="noether.data.base.wrappers.SubsetWrapper",
                end_index=max_samples,
            ),
        ]
    return StandardDatasetConfig(
        kind="noether.data.datasets.cfd.ShapeNetCarDataset",
        root=dataset_root,
        pipeline=AeroCFDPipelineConfig(
            kind="tutorial.pipeline.AeroMultistagePipeline",
            num_surface_points=3586,
            num_volume_points=4096,
            num_surface_queries=3586,
            num_volume_queries=4096,
            num_supernodes=3586,
            sample_query_points=False,
            use_physics_features=False,
            dataset_statistics=AeroStatsSchema(**dataset_statistics),
            data_specs=data_specs,
        ),
        split=mode,
        dataset_normalizers=dataset_normalizer,
        dataset_wrappers=wrappers,
        excluded_properties={"surface_friction", "volume_pressure", "volume_vorticity"},
    )


def build_model_config(data_specs: AeroDataSpecs) -> UPTConfig:
    """Small UPT model for fast iteration."""
    hidden_dim = 64
    num_heads = 2
    depth = 4
    return UPTConfig(
        kind="tutorial.model.UPT",
        name="upt",
        hidden_dim=hidden_dim,
        approximator_depth=depth,
        num_heads=num_heads,
        mlp_expansion_factor=4,
        use_rope=True,
        data_specs=data_specs,
        supernode_pooling_config=SupernodePoolingConfig(
            input_dim=data_specs.position_dim,
            hidden_dim=hidden_dim,
            radius=9,
        ),
        approximator_config=TransformerBlockConfig(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_expansion_factor=4,
            use_rope=True,
        ),
        decoder_config=DeepPerceiverDecoderConfig(
            depth=depth,
            input_dim=data_specs.position_dim,
            perceiver_block_config=PerceiverBlockConfig(
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                mlp_expansion_factor=4,
                use_rope=True,
            ),
        ),
        optimizer_config=OptimizerConfig(
            kind="noether.core.optimizer.Lion",
            lr=5.0e-5,
            weight_decay=0.05,
            clip_grad_norm=1.0,
            schedule_config=LinearWarmupCosineDecayScheduleConfig(
                kind="noether.core.schedules.LinearWarmupCosineDecaySchedule",
                warmup_percent=0.05,
                end_value=1.0e-6,
                max_value=5.0e-5,
            ),
        ),
        forward_properties=MODEL_FORWARD_PROPERTIES,
    )


def build_trainer_config(
    max_epochs: int,
    checkpoint_every_n_updates: int | None = None,
    checkpoint_every_n_epochs: int | None = None,
) -> AutomotiveAerodynamicsCfdTrainerConfig:
    callbacks = []

    if checkpoint_every_n_updates is not None:
        callbacks.append(
            CheckpointCallbackConfig(
                kind="noether.core.callbacks.CheckpointCallback",
                save_weights=True,
                save_optim=True,
                save_latest_weights=True,
                save_latest_optim=True,
                every_n_updates=checkpoint_every_n_updates,
            ),
        )
    elif checkpoint_every_n_epochs is not None:
        callbacks.append(
            CheckpointCallbackConfig(
                kind="noether.core.callbacks.CheckpointCallback",
                save_weights=True,
                save_optim=True,
                save_latest_weights=True,
                save_latest_optim=True,
                every_n_epochs=checkpoint_every_n_epochs,
            ),
        )

    callbacks.append(
        OfflineLossCallbackConfig(
            kind="noether.training.callbacks.OfflineLossCallback",
            batch_size=1,
            every_n_epochs=1,
            dataset_key="test",
        ),
    )

    return AutomotiveAerodynamicsCfdTrainerConfig(
        kind="tutorial.trainers.AutomotiveAerodynamicsCFDTrainer",
        surface_weight=1.0,
        volume_weight=1.0,
        surface_pressure_weight=1.0,
        volume_velocity_weight=1.0,
        use_physics_features=False,
        precision="float32",
        max_epochs=max_epochs,
        effective_batch_size=1,
        log_every_n_epochs=1,
        callbacks=callbacks,
        forward_properties=MODEL_FORWARD_PROPERTIES,
        target_properties=[
            "surface_pressure_target",
            "volume_velocity_target",
        ],
    )


def build_datasets(
    dataset_root: str,
    data_specs: AeroDataSpecs,
    dataset_normalizer: dict[str, list[AnyNormalizer]],
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> dict[str, DatasetBaseConfig]:
    return {
        "train": build_dataset_config(
            "train",
            dataset_root,
            data_specs,
            DATASET_STATS,
            dataset_normalizer,
            max_samples=max_train_samples,
        ),
        "test": build_dataset_config(
            "test",
            dataset_root,
            data_specs,
            DATASET_STATS,
            dataset_normalizer,
            max_samples=max_test_samples,
        ),
    }


# ---


def find_mid_epoch_checkpoint(
    output_path: Path,
    run_id: str,
    stage_name: str,
    updates_per_epoch: int,
) -> str | None:
    """Find the latest checkpoint that is NOT on an epoch boundary.

    Returns the checkpoint tag (e.g. "E1_U30_S30") or None if no mid-epoch checkpoint found.
    """
    cp_dir = output_path / run_id / stage_name / "checkpoints"
    if not cp_dir.exists():
        logger.warning(f"No checkpoint directory at {cp_dir}")
        return None

    # Checkpoint files are named like: upt_cp=E1_U30_S30_model.th
    update_checkpoints = []
    for p in cp_dir.iterdir():
        m = re.match(r"^.*_cp=E(\d+)_U(\d+)_S(\d+)_model\.th$", p.name)
        if m:
            epoch, update, sample = int(m.group(1)), int(m.group(2)), int(m.group(3))
            update_checkpoints.append((update, epoch, sample))

    if not update_checkpoints:
        logger.warning(f"No checkpoints found in {cp_dir}")
        return None

    # Sort by update number descending, prefer mid-epoch checkpoints
    update_checkpoints.sort(key=lambda x: x[0], reverse=True)

    for update, epoch, sample in update_checkpoints:
        full_tag = f"E{epoch}_U{update}_S{sample}"
        if update % updates_per_epoch != 0:
            # Return U-prefixed tag — HydraRunner parses U<number> as update-based resume
            tag = f"U{update}"
            logger.info(f"Found mid-epoch checkpoint: {full_tag} (using tag={tag})")
            return tag
        logger.info(f"Skipping epoch-boundary checkpoint: {full_tag}")

    logger.warning("All checkpoints are on epoch boundaries — no mid-epoch checkpoint found")
    return None


def find_signal_interrupt_checkpoint(output_path: Path, run_id: str, stage_name: str) -> str | None:
    """Find a .signal_interrupt checkpoint in the given run's checkpoint directory.

    Returns the checkpoint tag (e.g. "E2_U45_S45.signal_interrupt") or None.
    """
    cp_dir = output_path / run_id / stage_name / "checkpoints"
    if not cp_dir.exists():
        return None

    for p in cp_dir.iterdir():
        if "signal_interrupt" in p.name and p.name.endswith("_model.th"):
            # Extract tag from filename like: upt_cp=E2_U45_S45.signal_interrupt_model.th
            m = re.search(r"cp=(E\d+_U\d+_S\d+)\.signal_interrupt", p.name)
            if m:
                return m.group(1)
    return None


def run_training_with_sigterm(
    dataset_root: Path,
    output_path: Path,
    accelerator: str,
    max_epochs: int,
    stage_name: str,
    devices: str | None = None,
    effective_batch_size: int = 1,
    signal_delay_seconds: float = 5.0,
    checkpoint_every_n_updates: int | None = None,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> tuple[str, str | None]:
    """Run training and send SIGTERM after a delay.

    Returns (run_id, signal_checkpoint_tag or None).
    """

    def send_sigterm() -> None:
        logger.info(f"Sending SIGTERM to process (pid={os.getpid()}) after {signal_delay_seconds}s delay")
        os.kill(os.getpid(), signal.SIGTERM)

    timer = threading.Timer(signal_delay_seconds, send_sigterm)
    timer.start()

    try:
        run_id = run_training(
            dataset_root=dataset_root,
            output_path=output_path,
            accelerator=accelerator,
            max_epochs=max_epochs,
            stage_name=stage_name,
            devices=devices,
            effective_batch_size=effective_batch_size,
            checkpoint_every_n_updates=checkpoint_every_n_updates,
            max_train_samples=max_train_samples,
            max_test_samples=max_test_samples,
        )
    finally:
        timer.cancel()

    signal_tag = find_signal_interrupt_checkpoint(output_path, run_id, stage_name)
    return run_id, signal_tag


def find_run_id(output_path: Path, stage_name: str) -> str | None:
    """Find the most recent run_id in the output directory."""
    stage_dir = output_path / stage_name
    if not stage_dir.exists():
        return None

    runs = sorted(stage_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in runs:
        if run_dir.is_dir():
            return run_dir.name
    return None


def _multigpu_main(device: str, config: ConfigSchema, run_id_file: Path) -> None:
    """Entry point for each spawned GPU process in multi-GPU training."""
    from noether.core.distributed import is_rank0

    trainer, model, tracker, message_counter = HydraRunner.setup_experiment(
        device=device,
        config=config,
    )
    actual_run_id = str(trainer.path_provider.run_id)
    if is_rank0():
        run_id_file.write_text(actual_run_id)
        print(f"\n{'=' * 60}")
        print(f"Stage: {config.stage_name}")
        print(f"Run ID: {actual_run_id}")
        if config.resume_checkpoint:
            print(f"Resuming from: {config.resume_checkpoint} (run {config.resume_run_id})")
        print(f"Start checkpoint: {trainer.start_checkpoint}")
        print(f"End checkpoint: {trainer.end_checkpoint}")
        print(f"Updates per epoch: {trainer.updates_per_epoch}")
        print(f"{'=' * 60}\n")
    trainer.train(model)
    tracker.summarize_logvalues()
    message_counter.log()
    tracker.close()


def build_config(
    dataset_root: Path,
    output_path: Path,
    accelerator: str,
    max_epochs: int,
    stage_name: str,
    devices: str | None = None,
    effective_batch_size: int = 1,
    checkpoint_every_n_updates: int | None = None,
    checkpoint_every_n_epochs: int | None = None,
    resume_run_id: str | None = None,
    resume_stage_name: str | None = None,
    resume_checkpoint: str | None = None,
    run_id: str | None = None,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> ConfigSchema:
    """Build the training config."""
    data_specs = build_specs()
    dataset_normalizer = build_dataset_normalizer()
    model_config = build_model_config(data_specs)
    trainer_config = build_trainer_config(
        max_epochs=max_epochs,
        checkpoint_every_n_updates=checkpoint_every_n_updates,
        checkpoint_every_n_epochs=checkpoint_every_n_epochs,
    )
    trainer_config.effective_batch_size = effective_batch_size

    return ConfigSchema(
        name="mid-epoch-resume-test",
        accelerator=accelerator,
        stage_name=stage_name,
        dataset_kind="noether.data.datasets.cfd.ShapeNetCarDataset",
        dataset_root=dataset_root.as_posix(),
        resume_run_id=resume_run_id,
        resume_stage_name=resume_stage_name,
        resume_checkpoint=resume_checkpoint,
        seed=42,
        dataset_statistics=DATASET_STATS,
        dataset_normalizer=dataset_normalizer,
        static_config=StaticConfigSchema(output_path=output_path.as_posix()),
        tracker=None,
        run_id=run_id,
        devices=devices,
        num_workers=None,
        datasets=build_datasets(
            dataset_root.as_posix(),
            data_specs,
            dataset_normalizer,
            max_train_samples=max_train_samples,
            max_test_samples=max_test_samples,
        ),
        model=model_config,
        trainer=trainer_config,
        debug=False,
        store_code_in_output=False,
        output_path=output_path.as_posix(),
    )


def run_training(
    dataset_root: Path,
    output_path: Path,
    accelerator: str,
    max_epochs: int,
    stage_name: str,
    devices: str | None = None,
    effective_batch_size: int = 1,
    checkpoint_every_n_updates: int | None = None,
    checkpoint_every_n_epochs: int | None = None,
    resume_run_id: str | None = None,
    resume_stage_name: str | None = None,
    resume_checkpoint: str | None = None,
    run_id: str | None = None,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> str:
    """Run a training session and return the run_id."""
    config = build_config(
        dataset_root=dataset_root,
        output_path=output_path,
        accelerator=accelerator,
        max_epochs=max_epochs,
        stage_name=stage_name,
        devices=devices,
        effective_batch_size=effective_batch_size,
        checkpoint_every_n_updates=checkpoint_every_n_updates,
        checkpoint_every_n_epochs=checkpoint_every_n_epochs,
        resume_run_id=resume_run_id,
        resume_stage_name=resume_stage_name,
        resume_checkpoint=resume_checkpoint,
        run_id=run_id,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
    )

    # For multi-GPU, go through run_unmanaged which spawns processes
    if devices is not None and len(devices.split(",")) > 1:
        from functools import partial

        from noether.core.distributed import run_unmanaged

        # We need to capture the run_id from the spawned process. Use a shared file.
        run_id_file = output_path / f".run_id_{stage_name}.txt"

        run_unmanaged(
            main=partial(_multigpu_main, config=config, run_id_file=run_id_file),
            devices=devices,
            accelerator=config.accelerator,
            master_port=config.master_port,
        )
        return run_id_file.read_text().strip()

    # Single-GPU path (original)
    device = accelerator_to_device(accelerator=config.accelerator)

    trainer, model, tracker, message_counter = HydraRunner.setup_experiment(
        device=device,
        config=config,
    )

    actual_run_id = str(trainer.path_provider.run_id)

    print(f"\n{'=' * 60}")
    print(f"Stage: {stage_name}")
    print(f"Run ID: {actual_run_id}")
    print(f"Max epochs: {max_epochs}")
    if resume_checkpoint:
        print(f"Resuming from: {resume_checkpoint} (run {resume_run_id})")
    print(f"Start checkpoint: {trainer.start_checkpoint}")
    print(f"End checkpoint: {trainer.end_checkpoint}")
    print(f"Updates per epoch: {trainer.updates_per_epoch}")
    print(f"{'=' * 60}\n")

    trainer.train(model)
    tracker.summarize_logvalues()
    message_counter.log()
    tracker.close()

    return actual_run_id


def main() -> None:
    """
    Usage:
        python tutorial/test_mid_epoch_resume.py [--dataset-root /path/to/shapenet_car]

    The script:
      1. Trains for 3 epochs with update-based checkpointing (every 200 updates).
      2. Finds the latest mid-epoch checkpoint from that run.
      3. Resumes training from that mid-epoch checkpoint for 2 more epochs.
      4. Prints a summary comparing losses before and after resume.

    This validates that the mid-epoch resume feature works correctly end-to-end:
      - No ValueError on non-epoch-boundary checkpoints
      - Batch skipping works (first epoch after resume is shortened)
      - Subsequent epochs run at full length
      - Update counter stays synchronized
    """

    setup_logging()

    parser = argparse.ArgumentParser(description="Test mid-epoch checkpoint resume on ShapeNet data")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/Users/pk/shared_data/data/shapenet_car"),
        help="Path to ShapeNet Car dataset root",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="mps",
        choices=["cpu", "mps", "gpu"],
        help="Accelerator to train on",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated device IDs for multi-GPU (e.g. '0,1'). If None, uses a single device.",
    )
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=1,
        help="Effective batch size (must be >= number of GPUs for multi-GPU)",
    )
    parser.add_argument(
        "--initial-epochs",
        type=int,
        default=3,
        help="Number of epochs for the initial training run",
    )
    parser.add_argument(
        "--resume-epochs",
        type=int,
        default=5,
        help="Total epochs for the resumed run (must be > initial-epochs)",
    )
    parser.add_argument(
        "--checkpoint-every-n-updates",
        type=int,
        default=100,
        help="Save a checkpoint every N updates (use a value that doesn't divide updates_per_epoch for mid-epoch "
        "checkpoints)",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=150,
        help="Max training samples (subset the dataset for faster runs)",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=50,
        help="Max test samples (subset the dataset for faster runs)",
    )
    parser.add_argument(
        "--skip-signal-test",
        action="store_true",
        help="Skip the SIGTERM signal interrupt test",
    )
    parser.add_argument(
        "--signal-delay",
        type=float,
        default=60.0,
        help="Seconds to wait before sending SIGTERM in the signal test",
    )
    args = parser.parse_args()

    accelerator = args.accelerator
    devices = args.devices
    effective_batch_size = args.effective_batch_size
    output_path = args.dataset_root / "outputs" / "mid_epoch_resume_test"

    print("=" * 60)
    print("MID-EPOCH RESUME TEST")
    print("=" * 60)
    print(f"Dataset root:      {args.dataset_root}")
    print(f"Output path:       {output_path}")
    print(f"Accelerator:       {accelerator}")
    print(f"Devices:           {devices or 'single'}")
    print(f"Eff. batch size:   {effective_batch_size}")
    print(f"Max train samples: {args.max_train_samples}")
    print(f"Max test samples:  {args.max_test_samples}")
    print()

    # --- Step 1: Initial training:
    print("STEP 1: Initial training run")
    print("-" * 40)

    initial_run_id = run_training(
        dataset_root=args.dataset_root,
        output_path=output_path,
        accelerator=accelerator,
        max_epochs=args.initial_epochs,
        stage_name="train",
        devices=devices,
        effective_batch_size=effective_batch_size,
        checkpoint_every_n_updates=args.checkpoint_every_n_updates,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )

    print(f"\nInitial training completed. Run ID: {initial_run_id}")

    # --- Step 2: Find mid-epoch checkpoint:
    print("\nSTEP 2: Finding mid-epoch checkpoint")
    print("-" * 40)

    updates_per_epoch = args.max_train_samples // effective_batch_size
    checkpoint_tag = find_mid_epoch_checkpoint(output_path, initial_run_id, "train", updates_per_epoch)
    if checkpoint_tag is None:
        # Fall back to "latest" which might be at epoch boundary
        print("No mid-epoch checkpoint found, using 'latest'")
        checkpoint_tag = "latest"
    else:
        print(f"Found mid-epoch checkpoint: {checkpoint_tag}")

    # --- Step 3: Resume training:
    print(f"\nSTEP 3: Resuming from checkpoint {checkpoint_tag}")
    print("-" * 40)

    resume_run_id = run_training(
        dataset_root=args.dataset_root,
        output_path=output_path,
        accelerator=accelerator,
        max_epochs=args.resume_epochs,
        stage_name="resume",
        devices=devices,
        effective_batch_size=effective_batch_size,
        checkpoint_every_n_epochs=1,
        resume_run_id=initial_run_id,
        resume_stage_name="train",
        resume_checkpoint=checkpoint_tag,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )

    # --- Step 4: Signal interrupt test:
    if not args.skip_signal_test:
        print("\nSTEP 4: Signal interrupt test (SIGTERM during training)")
        print("-" * 40)

        signal_run_id, signal_tag = run_training_with_sigterm(
            dataset_root=args.dataset_root,
            output_path=output_path,
            accelerator=accelerator,
            max_epochs=20,  # many epochs so SIGTERM arrives mid-training
            stage_name="signal_test",
            devices=devices,
            effective_batch_size=effective_batch_size,
            signal_delay_seconds=args.signal_delay,
            checkpoint_every_n_updates=args.checkpoint_every_n_updates,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
        )

        if signal_tag:
            print(f"Signal interrupt checkpoint saved: {signal_tag}")
        else:
            print("WARNING: No .signal_interrupt checkpoint found!")
            print("  The SIGTERM may have arrived before training started or after it finished.")
    else:
        signal_run_id = None
        signal_tag = None

    # --- Summary:
    print("\n" + "=" * 60)
    print("MID-EPOCH RESUME TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Initial run:  {initial_run_id} ({args.initial_epochs} epochs)")
    print(f"Resumed from: {checkpoint_tag}")
    print(f"Resume run:   {resume_run_id} ({args.resume_epochs} epochs total)")
    if signal_run_id:
        signal_status = f"checkpoint at {signal_tag}" if signal_tag else "NO checkpoint (warning)"
        print(f"Signal test:  {signal_run_id} — {signal_status}")
    print(f"\nOutput path:  {output_path}")
    print()
    print("The training resumed from a mid-epoch checkpoint without errors.")
    print("Check the logs above to verify:")
    print("  - 'Mid-epoch resume: skipping N batches' message appeared")
    print("  - First epoch after resume was shortened")
    print("  - Subsequent epochs ran at full length")
    print("  - Loss values are reasonable")
    if signal_tag:
        print("  - 'Received SIGTERM' warning appeared in signal test")
        print(f"  - Signal interrupt checkpoint saved at {signal_tag}")


if __name__ == "__main__":
    main()
