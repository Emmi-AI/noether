#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""experiment factory functions for AB-UPT regression and data-space diffusion.

Shared building blocks (``DRIVAERML_FIELD_WEIGHTS``, ``ABUPT_FORWARD_PROPERTIES``,
``_build_schedule_config``) stay here and are imported by the sibling modules.
"""

from __future__ import annotations

from typing import Any

from aero_cfd.presets import DrivAerMLPreset

from noether.core.schemas.callbacks import EmaCallbackConfig
from noether.core.schemas.schema import ConfigSchema
from noether.modeling.diffusion import AnyDiffusionScheduleConfig, FlowMatchingConfig
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


def build_abupt_config(
    dataset_root: str,
    paradigm: str = "flow_matching",
    output_path: str = "./outputs/abupt",
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
    eval_sampling_steps: int = 5,
    chunked_eval_repetitions: int = 10,
    chunked_eval_num_surface_points: int = 1_000_000_000,
    chunked_eval_num_volume_points: int = 1_000_000_000,
    ema_decays: list[float] | None = None,
    ema_save_every_n_epochs: int = 10,
    minibatch_ot: bool = False,
    **kwargs: Any,
) -> ConfigSchema:
    """Build AB-UPT config for regression or data-space diffusion.

    Default hyperparameters match dataspace.md: hidden_dim=192, num_heads=3,
    9.1M params, 16K anchors, 65K geometry, 16K supernodes.

    Args:
        dataset_root: Root directory of the preprocessed DrivAerML dataset.
        paradigm: ``"regression"`` predicts fields directly from geometry
            (AeroABUPT + WeightedLossTrainer, built-in evaluation enabled).
            Any other value (e.g. ``"flow_matching"``) trains data-space
            diffusion (DiffusionABUPT + DiffusionABUPTTrainer) with EMA and
            chunked diffusion eval callbacks.

    Returns:
        Composed :class:`ConfigSchema` ready for ``HydraRunner``.
    """
    if physics_blocks is None:
        physics_blocks = ["perceiver", "self", "cross", "self", "cross", "self"]

    is_regression = paradigm == "regression"

    if is_regression:
        model_kind = "noether.modeling.models.aerodynamics.AeroABUPT"
        forward_properties = ABUPT_REGRESSION_FORWARD_PROPERTIES
    else:
        model_kind = "models.diffusion_abupt.DiffusionABUPT"
        forward_properties = ABUPT_FORWARD_PROPERTIES

    preset = DrivAerMLPreset()
    preset.forward_properties_map[model_kind] = forward_properties
    preset.pipeline_model_overrides[model_kind] = {
        "num_geometry_supernodes": num_geometry_supernodes,
        "num_geometry_points": num_geometry_points,
        "num_surface_anchor_points": num_surface_anchor_points,
        "num_volume_anchor_points": num_volume_anchor_points,
    }

    from noether.core.schemas.modules.blocks import TransformerBlockConfig
    from noether.core.schemas.modules.encoders import SupernodePoolingConfig as SPConfig

    spool_cfg = SPConfig(hidden_dim=hidden_dim, input_dim=3, radius=supernode_radius, bias=False)
    block_kwargs: dict[str, Any] = dict(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        mlp_expansion_factor=mlp_expansion_factor,
        use_rope=True,
        bias=False,
        attention_arguments={"qk_norm": True},
        max_wavelength=40_000,
    )

    block_cfg = TransformerBlockConfig(**block_kwargs)

    extra_callbacks: list[Any] = (
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
    )

    model_args = dict(
        hidden_dim=hidden_dim,
        supernode_pooling_config=spool_cfg,
        transformer_block_config=block_cfg,
        geometry_depth=geometry_depth,
        physics_blocks=physics_blocks,
        num_domain_decoder_blocks={
            "surface": num_surface_blocks,
            "volume": num_volume_blocks,
        },
    )

    from noether.core.schemas.dataset import RepeatWrapperConfig

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

    extra_datasets = {"chunked_test": chunked_test_ds}

    if is_regression:
        model_params: dict[str, Any] = dict(name="abupt_regression", **model_args)
        trainer_kind = "noether.training.trainers.WeightedLossTrainer"
        trainer_params: dict[str, Any] = dict(field_weights=DRIVAERML_FIELD_WEIGHTS)
    else:
        schedule_config = _build_schedule_config(paradigm)
        schedule_config.minibatch_ot = minibatch_ot  # override from CLI flag

        model_params = dict(
            name="diffusion_ab_upt",
            condition_dim=time_embed_dim,
            **model_args,
        )
        trainer_kind = "trainer.diffusion_ab_upt_trainer.DiffusionABUPTTrainer"
        trainer_params = dict(
            schedule_config=schedule_config,
            precision=precision,
            monitor_training_stability=True,
        )

        from callbacks.dataspace_diffusion_chunked_eval import (
            DataspaceDiffusionChunkedEvalCallbackConfig,
        )

        extra_callbacks.extend(
            [
                # per-epoch sample-based eval on `test` (training-size anchors, one
                # pass). Logs loss/test/<field>_{mse,mae,l2err} denormalized — same
                # keys as the regression callback, so diffusion and regression runs
                # are directly comparable.
                DataspaceDiffusionChunkedEvalCallbackConfig(
                    kind="callbacks.dataspace_diffusion_chunked_eval.DataspaceDiffusionChunkedEvalCallback",
                    every_n_epochs=eval_every_n_epochs,
                    dataset_key="test",
                    forward_properties=forward_properties,
                    chunked_inference=False,
                    sampling_steps=[eval_sampling_steps],
                    schedule_config=schedule_config,
                ),
                # full-mesh chunked eval at end of training only. Logs
                # loss/chunked_test/<field>_{mse,mae,l2err} denormalized.
                DataspaceDiffusionChunkedEvalCallbackConfig(
                    kind="callbacks.dataspace_diffusion_chunked_eval.DataspaceDiffusionChunkedEvalCallback",
                    every_n_epochs=max_epochs,  # end-of-training only — expensive
                    dataset_key="chunked_test",
                    forward_properties=forward_properties,
                    chunked_inference=True,
                    chunk_properties=["surface_anchor_position", "volume_anchor_position"],
                    chunk_size=num_surface_anchor_points,
                    sample_size_property="surface_anchor_position",
                    sampling_steps=[eval_sampling_steps],
                    schedule_config=schedule_config,
                ),
            ]
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
        model_params=model_params,
        trainer_kind=trainer_kind,
        trainer_params=trainer_params,
        dataset_root=dataset_root,
        output_path=output_path,
        datasets=["train", "val", "test"],
        extra_datasets=extra_datasets,
        max_epochs=max_epochs,
        batch_size=batch_size,
        include_evaluation=is_regression,
        extra_callbacks=extra_callbacks,
        chunk_size=num_surface_anchor_points,
        **kwargs,
    )


def run_diffusion_ab_upt(dataset_root: str, device: str = "cuda", **kwargs: Any) -> None:
    config = build_abupt_config(dataset_root=dataset_root, paradigm="flow_matching", **kwargs)
    HydraRunner.main(device=device, config=config)
