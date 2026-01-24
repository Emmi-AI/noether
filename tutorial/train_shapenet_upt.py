#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import torch

from noether.core.schemas.callbacks import (
    BestCheckpointCallbackConfig,
    CheckpointCallbackConfig,
    EmaCallbackConfig,
    OfflineLossCallbackConfig,
)
from noether.core.schemas.dataset import AeroDataSpecs, DatasetBaseConfig, DatasetWrappers, RepeatWrapperConfig
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
from tutorial.callbacks.surface_volume_evaluation_metrics import (
    SurfaceVolumeEvaluationMetricsCallbackConfig,
)
from tutorial.schemas.models.upt_config import UPTConfig
from tutorial.schemas.pipelines.aero_pipeline_config import AeroCFDPipelineConfig
from tutorial.schemas.trainers.automotive_aerodynamics_trainer_config import AutomotiveAerodynamicsCfdTrainerConfig

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
    "surface_feature_dim": {
        "surface_sdf": 1,
        "surface_normals": 3,
    },
    "volume_feature_dim": {
        "volume_sdf": 1,
        "volume_normals": 3,
    },
    "surface_output_dims": {
        "pressure": 1,
    },
    "volume_output_dims": {
        "velocity": 3,
    },
}


def build_stats() -> AeroStatsSchema:
    return AeroStatsSchema(**DATASET_STATS)


def build_specs() -> AeroDataSpecs:
    return AeroDataSpecs(**DATA_SPECS)


def build_dataset_config(
    mode: Literal["train", "test"],
    dataset_root: str,
    data_specs: dict[str, Any] | AeroDataSpecs,
    dataset_statistics: dict[str, Sequence[float]],
    dataset_normalizer: dict[str, list[AnyNormalizer]],
    dataset_wrappers: list[DatasetWrappers] | None = None,
) -> DatasetBaseConfig:
    return DatasetBaseConfig(
        kind="noether.data.datasets.cfd.ShapeNetCarDataset",
        root=dataset_root,
        pipeline=AeroCFDPipelineConfig(
            kind="tutorial.pipeline.AeroMultistagePipeline",
            num_surface_points=3586,  # max = 3586
            num_volume_points=4096,  # max = 28504
            num_surface_queries=3586,
            num_volume_queries=4096,
            num_supernodes=3586,
            sample_query_points=False,
            use_physics_features=False,
            dataset_statistics=AeroStatsSchema(**dataset_statistics),
            data_specs=data_specs if isinstance(data_specs, AeroDataSpecs) else AeroDataSpecs(**data_specs),
        ),
        split=mode,
        dataset_normalizers=dataset_normalizer,
        dataset_wrappers=dataset_wrappers,
        included_properties=None,
        excluded_properties={"surface_friction", "volume_pressure", "volume_vorticity"},
    )


def build_dataset_normalizer() -> dict[str, list[AnyNormalizer]]:
    dataset_normalizer: dict[str, list[AnyNormalizer]] = {
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
            )
        ],
    }
    return dataset_normalizer


def build_model_config(
    data_specs: AeroDataSpecs,
    model_forward_properties: list[str],
) -> UPTConfig:
    model_config = UPTConfig(
        kind="tutorial.model.UPT",
        name="upt",
        hidden_dim=192,
        approximator_depth=12,
        num_heads=3,
        mlp_expansion_factor=4,
        use_rope=True,
        data_specs=data_specs,
        supernode_pooling_config=SupernodePoolingConfig(
            input_dim=data_specs.position_dim,
            hidden_dim=192,
            radius=9,
        ),
        approximator_config=TransformerBlockConfig(
            num_heads=3,
            hidden_dim=192,
            mlp_expansion_factor=4,
            use_rope=True,
        ),
        decoder_config=DeepPerceiverDecoderConfig(
            depth=12,
            input_dim=data_specs.position_dim,
            perceiver_block_config=PerceiverBlockConfig(
                num_heads=3,
                hidden_dim=192,
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
        forward_properties=model_forward_properties,
    )
    return model_config


def build_trainer_config(model_forward_properties: list[str]) -> AutomotiveAerodynamicsCfdTrainerConfig:
    batch_size = 1
    loss_and_log_every_n_epochs = 1
    save_and_ema_every_n_epochs = 10

    return AutomotiveAerodynamicsCfdTrainerConfig(
        kind="tutorial.trainers.AutomotiveAerodynamicsCFDTrainer",
        surface_weight=1.0,
        volume_weight=1.0,
        surface_pressure_weight=1.0,
        volume_velocity_weight=1.0,
        use_physics_features=False,
        precision="float32",
        max_epochs=500,
        effective_batch_size=batch_size,
        log_every_n_epochs=loss_and_log_every_n_epochs,
        callbacks=[
            CheckpointCallbackConfig(
                kind="noether.core.callbacks.CheckpointCallback",
                save_weights=True,
                save_latest_weights=True,
                save_latest_optim=False,
                every_n_epochs=save_and_ema_every_n_epochs,
            ),
            # validation loss
            OfflineLossCallbackConfig(
                kind="noether.training.callbacks.OfflineLossCallback",
                batch_size=batch_size,
                every_n_epochs=loss_and_log_every_n_epochs,
                dataset_key="test",
            ),
            BestCheckpointCallbackConfig(
                kind="noether.core.callbacks.BestCheckpointCallback",
                every_n_epochs=batch_size,
                metric_key="loss/test/total",
            ),
            # test loss
            SurfaceVolumeEvaluationMetricsCallbackConfig(
                kind="tutorial.callbacks.SurfaceVolumeEvaluationMetricsCallback",
                batch_size=batch_size,
                every_n_epochs=loss_and_log_every_n_epochs,
                dataset_key="test",
                forward_properties=model_forward_properties,
            ),
            SurfaceVolumeEvaluationMetricsCallbackConfig(
                kind="tutorial.callbacks.SurfaceVolumeEvaluationMetricsCallback",
                batch_size=batch_size,
                every_n_epochs=500,
                dataset_key="test_repeat",
                forward_properties=model_forward_properties,
            ),
            # ema
            EmaCallbackConfig(
                kind="noether.core.callbacks.EmaCallback",
                every_n_epochs=save_and_ema_every_n_epochs,
                save_weights=False,
                save_last_weights=False,
                save_latest_weights=True,
                target_factors={0.9999},
            ),
        ],
        forward_properties=model_forward_properties,
        target_properties=[
            "surface_pressure_target",
            "volume_velocity_target",
        ],
    )


def main() -> None:
    dataset_root = Path("/Users/pk/shared_data/data/shapenet_car")
    output_path = dataset_root / "outputs"

    data_specs = build_specs()
    dataset_normalizer = build_dataset_normalizer()

    model_forward_properties = [
        "surface_mask_query",
        "surface_position_batch_idx",
        "surface_position_supernode_idx",
        "surface_position",
        "surface_query_position",
        "volume_query_position",
    ]

    upt_model_config = build_model_config(data_specs, model_forward_properties)
    aero_trainer_config = build_trainer_config(model_forward_properties)

    HydraRunner().main(
        device=torch.device("mps"),
        config=ConfigSchema(
            name=None,
            accelerator="mps",  # can be "cpu", "gpu", "mps"
            stage_name="train",
            dataset_kind="noether.data.datasets.cfd.ShapeNetCarDataset",
            dataset_root=dataset_root.as_posix(),
            resume_run_id=None,
            resume_stage_name=None,
            resume_checkpoint=None,
            seed=42,
            dataset_statistics=DATASET_STATS,
            dataset_normalizer=dataset_normalizer,
            static_config=StaticConfigSchema(output_path=output_path.as_posix()),
            tracker=None,
            run_id=None,
            devices=None,
            num_workers=None,
            datasets={
                "train": build_dataset_config(
                    mode="train",
                    dataset_root=dataset_root.as_posix(),
                    data_specs=data_specs,
                    dataset_statistics=DATASET_STATS,
                    dataset_normalizer=dataset_normalizer,
                    dataset_wrappers=None,
                ),
                "test": build_dataset_config(
                    mode="test",
                    dataset_root=dataset_root.as_posix(),
                    data_specs=data_specs,
                    dataset_statistics=DATASET_STATS,
                    dataset_normalizer=dataset_normalizer,
                    dataset_wrappers=None,
                ),
                "test_repeat": build_dataset_config(
                    mode="test",
                    dataset_root=dataset_root.as_posix(),
                    data_specs=data_specs,
                    dataset_statistics=DATASET_STATS,
                    dataset_normalizer=dataset_normalizer,
                    dataset_wrappers=[
                        RepeatWrapperConfig(
                            kind="noether.data.base.wrappers.RepeatWrapper",
                            repetitions=10,
                        )
                    ],
                ),
            },
            model=upt_model_config,
            trainer=aero_trainer_config,
            debug=False,
            store_code_in_output=False,
            output_path=output_path.as_posix(),
        ),
    )


if __name__ == "__main__":
    main()
