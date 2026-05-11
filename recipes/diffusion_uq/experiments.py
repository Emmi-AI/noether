#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""experiment factory functions for AB-UPT regression and data-space diffusion.

Other experiment families live in sibling modules:

- :mod:`autoencoder_experiments` — AB-UPT autoencoder pretraining.
- :mod:`latent_diffusion_experiments` — denoiser on top of a frozen AE.

Shared building blocks (``DRIVAERML_FIELD_WEIGHTS``, ``ABUPT_FORWARD_PROPERTIES``,
``_build_schedule_config``) stay here and are imported by the sibling modules.
"""

from __future__ import annotations

from typing import Any

from aero_cfd.presets import DrivAerMLPreset

from noether.core.schemas.diffusion import (
    AnyDiffusionScheduleConfig,
    FlowMatchingConfig,
)
from noether.core.schemas.schema import ConfigSchema
from noether.training.runners import HydraRunner


def _build_schedule_config(paradigm: str) -> AnyDiffusionScheduleConfig:
    """Resolve a free-form ``paradigm`` string to a default schedule config."""
    if paradigm == "flow_matching":
        return FlowMatchingConfig()
    raise ValueError(f"Unknown diffusion paradigm: {paradigm!r}")


DRIVAERML_FIELD_WEIGHTS = {
    "surface_pressure": 1.0,
    "surface_friction": 1.0,
    "volume_pressure": 1.0,
    "volume_velocity": 1.0,
    "volume_vorticity": 1.0,
}


ABUPT_FORWARD_PROPERTIES = [
    "geometry_position",
    "geometry_supernode_idx",
    "geometry_batch_idx",
    "surface_anchor_position",
    "volume_anchor_position",
    "surface_pressure_target",
    "surface_friction_target",
    "volume_pressure_target",
    "volume_velocity_target",
    "volume_vorticity_target",
]

ABUPT_DEFAULT_PIPELINE = {
    "num_geometry_supernodes": 16384,
    "num_geometry_points": 65536,
    "num_surface_anchor_points": 1024,
    "num_volume_anchor_points": 1024,
}

ABUPT_REGRESSION_FORWARD_PROPERTIES = [
    "geometry_position",
    "geometry_supernode_idx",
    "geometry_batch_idx",
    "surface_anchor_position",
    "volume_anchor_position",
]


def build_abupt_regression_config(
    dataset_root: str,
    output_path: str = "./outputs/abupt_regression",
    hidden_dim: int = 192,
    num_heads: int = 3,
    mlp_expansion_factor: int = 4,
    geometry_depth: int = 1,
    physics_blocks: list[str] | None = None,
    num_surface_blocks: int = 6,
    num_volume_blocks: int = 6,
    num_geometry_supernodes: int = 16384,
    num_geometry_points: int = 65536,
    num_surface_anchor_points: int = 1024,
    num_volume_anchor_points: int = 1024,
    supernode_radius: float = 0.25,
    max_epochs: int = 500,
    batch_size: int = 1,
    lr: float = 5e-5,
    warmup_percent: float = 0.05,
    end_lr: float | None = 1e-6,
    weight_decay: float = 0.05,
    clip_grad_norm: float | None = 1.0,
    eval_every_n_epochs: int = 1,
    **kwargs: Any,
) -> ConfigSchema:
    """Build regression AB-UPT config: predicts fields directly from geometry.

    Uses noether's built-in AeroABUPT (no latent bottleneck, no field injection).
    Strict baseline for the diffusion model.
    """
    if physics_blocks is None:
        physics_blocks = ["perceiver", "self", "cross", "self", "cross", "self"]

    model_kind = "noether.modeling.models.aerodynamics.AeroABUPT"

    preset = DrivAerMLPreset()
    preset.forward_properties_map[model_kind] = ABUPT_REGRESSION_FORWARD_PROPERTIES
    preset.pipeline_model_overrides[model_kind] = {
        "num_geometry_supernodes": num_geometry_supernodes,
        "num_geometry_points": num_geometry_points,
        "num_surface_anchor_points": num_surface_anchor_points,
        "num_volume_anchor_points": num_volume_anchor_points,
    }

    from noether.core.schemas.modules.blocks import TransformerBlockConfig
    from noether.core.schemas.modules.encoders import SupernodePoolingConfig as SPConfig

    spool_cfg = SPConfig(hidden_dim=hidden_dim, input_dim=3, radius=supernode_radius, bias=False)
    block_cfg = TransformerBlockConfig(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        mlp_expansion_factor=mlp_expansion_factor,
        use_rope=True,
        bias=False,
        attention_arguments={"qk_norm": True},
    )

    return preset.build_config(
        model_kind=model_kind,
        optimizer=preset.build_optimizer(
            lr=lr,
            warmup_percent=warmup_percent,
            end_lr=end_lr,
            weight_decay=weight_decay,
            clip_grad_norm=clip_grad_norm,
        ),
        model_params=dict(
            name="abupt_regression",
            hidden_dim=hidden_dim,
            supernode_pooling_config=spool_cfg,
            transformer_block_config=block_cfg,
            geometry_depth=geometry_depth,
            physics_blocks=physics_blocks,
            num_surface_blocks=num_surface_blocks,
            num_volume_blocks=num_volume_blocks,
        ),
        trainer_kind="noether.training.trainers.WeightedLossTrainer",
        trainer_params=dict(field_weights=DRIVAERML_FIELD_WEIGHTS),
        dataset_root=dataset_root,
        output_path=output_path,
        datasets=["train", "val", "test"],
        max_epochs=max_epochs,
        batch_size=batch_size,
        callbacks_override=preset.standard_callbacks(
            log_every_n_epochs=eval_every_n_epochs,
            batch_size=batch_size,
        ),
        **kwargs,
    )


def build_diffusion_ab_upt_config(
    dataset_root: str,
    output_path: str = "./outputs/diffusion_ab_upt",
    paradigm: str = "flow_matching",
    hidden_dim: int = 192,
    num_heads: int = 3,
    mlp_expansion_factor: int = 4,
    geometry_depth: int = 1,
    physics_blocks: list[str] | None = None,
    num_surface_blocks: int = 6,
    num_volume_blocks: int = 6,
    time_embed_dim: int = 256,
    # mesh sampling (dataspace.md: 65K geometry, 16K supernodes, 16K anchors)
    num_geometry_supernodes: int = 16384,
    num_geometry_points: int = 65536,
    num_surface_anchor_points: int = 16384,
    num_volume_anchor_points: int = 16384,
    supernode_radius: float = 0.25,
    max_epochs: int = 500,
    batch_size: int = 1,
    lr: float = 5e-5,
    warmup_percent: float = 0.05,
    end_lr: float | None = 1e-6,
    weight_decay: float = 0.05,
    clip_grad_norm: float | None = 1.0,
    precision: str = "float32",
    eval_every_n_epochs: int = 5,
    eval_sampling_steps: int = 10,
    chunked_eval_repetitions: int = 10,
    chunked_eval_num_surface_points: int = 1_000_000_000,
    chunked_eval_num_volume_points: int = 1_000_000_000,
    ema_decays: list[float] | None = None,
    ema_save_every_n_epochs: int = 10,
    **kwargs: Any,
) -> ConfigSchema:
    """build config for data-space diffusion on ab-upt (dataspace.md setup).

    default hyperparameters match dataspace.md: hidden_dim=192, num_heads=3,
    9.1M params, 16K anchors, 65K geometry, 1K supernodes.
    """
    if physics_blocks is None:
        physics_blocks = ["perceiver", "self", "cross", "self", "cross", "self"]

    model_kind = "models.diffusion_abupt.DiffusionABUPT"

    preset = DrivAerMLPreset()
    preset.forward_properties_map[model_kind] = ABUPT_FORWARD_PROPERTIES
    preset.pipeline_model_overrides[model_kind] = {
        "num_geometry_supernodes": num_geometry_supernodes,
        "num_geometry_points": num_geometry_points,
        "num_surface_anchor_points": num_surface_anchor_points,
        "num_volume_anchor_points": num_volume_anchor_points,
    }

    from callbacks.dataspace_diffusion_chunked_eval import (
        DataspaceDiffusionChunkedEvalCallbackConfig,
    )

    from noether.core.schemas.callbacks import EmaCallbackConfig
    from noether.core.schemas.dataset import RepeatWrapperConfig
    from noether.core.schemas.modules.blocks import TransformerBlockConfig
    from noether.core.schemas.modules.encoders import SupernodePoolingConfig as SPConfig

    spool_cfg = SPConfig(hidden_dim=hidden_dim, input_dim=3, radius=supernode_radius, bias=False)
    block_cfg = TransformerBlockConfig(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        mlp_expansion_factor=mlp_expansion_factor,
        use_rope=True,
        max_wavelength=40_000,  # cover the whole geometry space for absolute time encoding
        bias=False,
        attention_arguments={"qk_norm": True},
    )

    # chunked_test dataset: full-mesh anchor pipeline (1e9 = take all available
    # mesh points as anchors), wrapped with RepeatWrapper so the one-shot eval
    # sees multiple draws per sample. The callback slices back to training-size
    # chunks via chunk_size = num_surface_anchor_points.
    chunked_test_ds = preset.build_dataset(
        split="test",
        root=dataset_root,
        model_kind=model_kind,
        wrappers=[
            RepeatWrapperConfig(
                kind="noether.data.base.wrappers.RepeatWrapper",
                repetitions=chunked_eval_repetitions,
            )
        ],
        num_geometry_supernodes=num_geometry_supernodes,
        num_geometry_points=num_geometry_points,
        num_surface_anchor_points=chunked_eval_num_surface_points,
        num_volume_anchor_points=chunked_eval_num_volume_points,
    )

    abupt_forward_props = preset.forward_properties_map[model_kind]

    return preset.build_config(
        model_kind=model_kind,
        optimizer=preset.build_optimizer(
            lr=lr,
            warmup_percent=warmup_percent,
            end_lr=end_lr,
            weight_decay=weight_decay,
            clip_grad_norm=clip_grad_norm,
        ),
        model_params=dict(
            name="diffusion_ab_upt",
            hidden_dim=hidden_dim,
            supernode_pooling_config=spool_cfg,
            transformer_block_config=block_cfg,
            geometry_depth=geometry_depth,
            physics_blocks=physics_blocks,
            num_domain_decoder_blocks={"surface": num_surface_blocks, "volume": num_volume_blocks},
            condition_dim=time_embed_dim,
        ),
        trainer_kind="trainer.diffusion_ab_upt_trainer.DiffusionABUPTTrainer",
        trainer_params=dict(
            schedule_config=_build_schedule_config(paradigm),
            precision=precision,
            monitor_training_stability=True,
        ),
        dataset_root=dataset_root,
        output_path=output_path,
        datasets=["train", "val", "test"],
        extra_datasets={"chunked_test": chunked_test_ds},
        max_epochs=max_epochs,
        batch_size=batch_size,
        include_evaluation=False,  # regression eval callback doesn't fit diffusion outputs
        extra_callbacks=[
            *(
                [
                    EmaCallbackConfig(
                        kind="noether.core.callbacks.checkpoint.ema.EmaCallback",
                        every_n_epochs=ema_save_every_n_epochs,
                        target_factors=list(ema_decays),
                        save_weights=False,
                        save_last_weights=True,
                        save_latest_weights=True,
                    )
                ]
                if ema_decays
                else []
            ),
            # per-epoch sample-based eval on `test` (training-size anchors, one
            # pass). Logs loss/test/<field>_{mse,mae,l2err} denormalized — same
            # keys as the regression callback, so diffusion and regression runs
            # are directly comparable.
            DataspaceDiffusionChunkedEvalCallbackConfig(
                kind="callbacks.dataspace_diffusion_chunked_eval.DataspaceDiffusionChunkedEvalCallback",
                every_n_epochs=eval_every_n_epochs,
                dataset_key="test",
                forward_properties=abupt_forward_props,
                chunked_inference=False,
                sampling_steps=eval_sampling_steps,
                schedule_config=_build_schedule_config(paradigm),
            ),
            # full-mesh chunked eval at end of training only. Logs
            # loss/chunked_test/<field>_{mse,mae,l2err} denormalized.
            DataspaceDiffusionChunkedEvalCallbackConfig(
                kind="callbacks.dataspace_diffusion_chunked_eval.DataspaceDiffusionChunkedEvalCallback",
                every_n_epochs=max_epochs,  # end-of-training only — expensive
                dataset_key="chunked_test",
                forward_properties=abupt_forward_props,
                chunked_inference=True,
                chunk_properties=["surface_anchor_position", "volume_anchor_position"],
                chunk_size=num_surface_anchor_points,
                sample_size_property="surface_anchor_position",
                sampling_steps=eval_sampling_steps,
                schedule_config=_build_schedule_config(paradigm),
            ),
        ],
        **kwargs,
    )


def run_diffusion_ab_upt(dataset_root: str, device: str = "cuda", **kwargs: Any) -> None:
    config = build_diffusion_ab_upt_config(dataset_root=dataset_root, **kwargs)
    HydraRunner.main(device=device, config=config)
