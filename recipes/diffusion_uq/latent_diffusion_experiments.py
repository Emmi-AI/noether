#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""experiment factory functions for latent diffusion (denoiser on top of a frozen AE)."""

from __future__ import annotations

from typing import Any

from aero_cfd.presets import DrivAerMLPreset
from experiments import _build_schedule_config

from noether.core.schemas.schema import ConfigSchema
from noether.training.runners import HydraRunner


def load_ae_checkpoint(model, checkpoint_path: str, device: str = "cuda"):
    """load a state_dict from an explicit checkpoint path into an AE module."""
    import torch

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    print(f"loaded ae checkpoint: {checkpoint_path}")


def build_latent_diffusion_config(
    latent_root: str,
    cfd_root: str,
    output_path: str = "./outputs/latent_diffusion",
    ae_config: ConfigSchema | None = None,
    paradigm: str = "flow_matching",
    latent_scale: float | None = None,
    hidden_dim: int = 256,
    latent_dim: int = 256,
    denoiser_depth: int = 6,
    denoiser_heads: int = 8,
    condition_dim: int = 1024,
    denoiser_use_rope: bool = False,
    num_surface_tokens: int = 0,
    adaln_zero_std: float | None = None,
    num_geometry_supernodes: int = 16384,
    num_geometry_points: int = 65536,
    num_surface_anchor_points: int = 1024,
    num_volume_anchor_points: int = 1024,
    supernode_radius: float = 0.25,
    max_epochs: int = 2000,
    batch_size: int = 8,
    lr: float = 1e-4,
    warmup_percent: float = 0.05,
    end_lr: float | None = 1e-6,
    weight_decay: float = 0.05,
    clip_grad_norm: float | None = 1.0,
    precision: str = "float32",
    save_checkpoints: bool = True,
    **kwargs: Any,
) -> ConfigSchema:
    """build config for latent diffusion training.

    Args:
        latent_root: directory where extracted latent ``.pt`` files live (one
            per sample, grouped by split).
        cfd_root: root of the underlying DrivAerML CFD dataset; the latent
            dataset wraps it to re-derive anchor positions, geometry tokens,
            and ground-truth field targets at consume time.
    """
    from models.latent_abupt import LatentDenoiserConfig, LatentDiffusionModelConfig
    from trainer.latent_diffusion_trainer import LatentDiffusionTrainerConfig

    from noether.core.schemas.callbacks import (
        BestCheckpointCallbackConfig,
        CheckpointCallbackConfig,
        OfflineLossCallbackConfig,
    )
    from noether.core.schemas.optimizers import OptimizerConfig
    from noether.core.schemas.schedules import LinearWarmupCosineDecayScheduleConfig
    from noether.core.schemas.schema import ConfigSchema

    autoencoder_config = None
    if ae_config is not None:
        autoencoder_config = ae_config.model.model_copy(deep=True)
        autoencoder_config.is_frozen = True
        autoencoder_config.optimizer_config = None
        autoencoder_config.initializers = None

    schedule = None
    if end_lr is not None:
        schedule = LinearWarmupCosineDecayScheduleConfig(
            max_value=lr,
            warmup_percent=warmup_percent,
            end_value=end_lr,
        )

    denoiser_config = LatentDenoiserConfig(
        kind="models.latent_abupt.LatentDenoiser",
        name="denoiser",
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_heads=denoiser_heads,
        depth=denoiser_depth,
        condition_dim=condition_dim,
        use_rope=denoiser_use_rope,
        num_surface_tokens=num_surface_tokens,
        adaln_zero_std=adaln_zero_std,
        optimizer_config=OptimizerConfig(
            kind="torch.optim.AdamW",
            lr=lr,
            weight_decay=weight_decay,
            clip_grad_norm=clip_grad_norm,
            schedule_config=schedule,
        ),
    )

    return ConfigSchema(
        model=LatentDiffusionModelConfig(
            kind="models.latent_abupt.LatentDiffusionModel",
            name="latent_diffusion",
            denoiser_config=denoiser_config,
            autoencoder_config=autoencoder_config,
        ),
        trainer=LatentDiffusionTrainerConfig(
            kind="trainer.latent_diffusion_trainer.LatentDiffusionTrainer",
            max_epochs=max_epochs,
            effective_batch_size=batch_size,
            precision=precision,
            schedule_config=_build_schedule_config(paradigm),
            latent_stats_path=f"{latent_root}/train_stats.pt",
            callbacks=[
                CheckpointCallbackConfig(
                    kind="noether.core.callbacks.checkpoint.checkpoint.CheckpointCallback",
                    every_n_epochs=10,
                    save_latest_weights=True,
                ),
                OfflineLossCallbackConfig(
                    kind="noether.training.callbacks.OfflineLossCallback",
                    every_n_epochs=10,
                    dataset_key="test",
                    batch_size=batch_size,
                ),
                BestCheckpointCallbackConfig(
                    kind="noether.core.callbacks.checkpoint.best_checkpoint.BestCheckpointCallback",
                    every_n_epochs=10,
                    metric_key="loss/test/total",
                    model_names=["denoiser"],
                    save_frozen_weights=False,
                ),
            ]
            if save_checkpoints
            else [],
            forward_properties=["latents", "supernode_positions"],
            target_properties=["latents"],
            # avoid log/track OnlineLossCallback clash on small jobs
            # (both default to every_n_updates=1 → duplicate "loss/online/total/U1")
            log_every_n_epochs=1,
            track_every_n_updates=1,
        ),
        datasets=_build_latent_datasets(
            latent_root=latent_root,
            cfd_root=cfd_root,
            num_geometry_supernodes=num_geometry_supernodes,
            num_geometry_points=num_geometry_points,
            num_surface_anchor_points=num_surface_anchor_points,
            num_volume_anchor_points=num_volume_anchor_points,
            supernode_radius=supernode_radius,
        ),
        output_path=output_path,
        seed=42,
        **kwargs,
    )


def _build_latent_datasets(
    *,
    latent_root: str,
    cfd_root: str,
    num_geometry_supernodes: int,
    num_geometry_points: int,
    num_surface_anchor_points: int,
    num_volume_anchor_points: int,
    supernode_radius: float,
) -> dict[str, Any]:
    """Construct LatentDatasetConfigs paired with the underlying DrivAerML config + AB-UPT pipeline."""
    from datasets.latent_dataset import LatentDatasetConfig
    from datasets.latent_diffusion_pipeline import LatentDiffusionPipelineConfig

    preset = DrivAerMLPreset()
    # ``preset.build_dataset`` returns a StandardDatasetConfig with the
    # AeroCFD pipeline attached. We strip the pipeline at LatentDataset
    # construction time (the latent dataset uses its own pipeline that adds
    # the latent collator on top of the same sample processors), but keep
    # everything else (kind, root, normalizers, excluded_properties).
    cfd_train = preset.build_dataset(
        split="train",
        root=cfd_root,
        model_kind="models.autoencoder_abupt.ABUPTAutoencoder",
        num_geometry_supernodes=num_geometry_supernodes,
        num_geometry_points=num_geometry_points,
        num_surface_anchor_points=num_surface_anchor_points,
        num_volume_anchor_points=num_volume_anchor_points,
    )
    cfd_test = cfd_train.model_copy(deep=True)
    cfd_test.split = "test"

    # AeroCFD sample processors care about num_*; supernode_radius is a
    # model-side knob (encoder pooling) that doesn't affect dataset sampling
    # but we keep it on the signature for API parity with the AE builder.
    del supernode_radius

    pipeline_config = LatentDiffusionPipelineConfig(
        dataset_statistics=cfd_train.pipeline.dataset_statistics,
        data_specs=cfd_train.pipeline.data_specs,
        num_surface_points=cfd_train.pipeline.num_surface_points,
        num_volume_points=cfd_train.pipeline.num_volume_points,
        num_surface_queries=cfd_train.pipeline.num_surface_queries,
        num_volume_queries=cfd_train.pipeline.num_volume_queries,
        num_geometry_supernodes=num_geometry_supernodes,
        num_geometry_points=num_geometry_points,
        num_surface_anchor_points=num_surface_anchor_points,
        num_volume_anchor_points=num_volume_anchor_points,
        use_physics_features=cfd_train.pipeline.use_physics_features,
        sample_query_points=cfd_train.pipeline.sample_query_points,
    )

    return {
        "train": LatentDatasetConfig(
            kind="datasets.latent_dataset.LatentDataset",
            latent_root=latent_root,
            cfd_dataset_config=cfd_train,
            split="train",
            pipeline=pipeline_config,
        ),
        "test": LatentDatasetConfig(
            kind="datasets.latent_dataset.LatentDataset",
            latent_root=latent_root,
            cfd_dataset_config=cfd_test,
            split="test",
            pipeline=pipeline_config.model_copy(deep=True),
        ),
    }


def run_latent_diffusion(latent_root: str, cfd_root: str, device: str = "cuda", **kwargs: Any) -> None:
    config = build_latent_diffusion_config(latent_root=latent_root, cfd_root=cfd_root, **kwargs)
    HydraRunner.main(device=device, config=config)
