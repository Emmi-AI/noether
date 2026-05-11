#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""experiment factory functions for AB-UPT autoencoder pretraining on DrivAerML."""

from __future__ import annotations

from typing import Any

from aero_cfd.presets import DrivAerMLPreset
from experiments import ABUPT_FORWARD_PROPERTIES, DRIVAERML_FIELD_WEIGHTS

from noether.core.schemas.schema import ConfigSchema
from noether.training.runners import HydraRunner


def build_abupt_ae_pretrain_config(
    dataset_root: str,
    output_path: str = "./outputs/abupt_ae_pretrain",
    hidden_dim: int = 192,
    num_heads: int = 3,
    mlp_expansion_factor: int = 4,
    geometry_depth: int = 1,
    physics_blocks: list[str] | None = None,
    num_surface_blocks: int = 6,
    num_volume_blocks: int = 6,
    latent_dim: int = 512,
    surface_field_dim: int = 4,
    volume_field_dim: int = 7,
    # mesh sampling (65K geometry, 16K supernodes, 1K+1K anchors = 2048 latent tokens)
    num_geometry_supernodes: int = 16384,
    num_geometry_points: int = 65536,
    num_surface_anchor_points: int = 1024,
    num_volume_anchor_points: int = 1024,
    supernode_radius: float = 0.25,
    query_ratio: float = 0.0,
    latent_num_surface_tokens: int | None = None,
    latent_num_volume_tokens: int | None = None,
    bottleneck_num_heads: int = 4,
    bottleneck_mode: str | None = None,
    max_epochs: int = 500,
    batch_size: int = 1,
    lr: float = 5e-5,
    warmup_percent: float = 0.05,
    end_lr: float | None = 1e-6,
    weight_decay: float = 0.05,
    clip_grad_norm: float | None = 1.0,
    eval_every_n_epochs: int = 1,
    precision: str = "float32",
    **kwargs: Any,
) -> ConfigSchema:
    """build config for AB-UPT autoencoder pretraining on drivaerml.

    same backbone as DiffusionABUPT (geometry encoder, physics blocks, decoder
    branches) with an explicit downproj/upproj latent bottleneck. field injection
    enables field-aware latent encoding for downstream latent diffusion.

    query_ratio > 0: during training, encode at (1-ratio) of anchor points,
    decode at ALL anchor points. forces position-independent latent encoding.

    default hyperparameters match dataspace.md: hidden_dim=192, num_heads=3.
    """
    if physics_blocks is None:
        physics_blocks = ["perceiver", "self", "cross", "self", "cross", "self"]

    model_kind = "models.autoencoder_abupt.ABUPTAutoencoder"

    preset = DrivAerMLPreset()
    preset.forward_properties_map[model_kind] = ABUPT_FORWARD_PROPERTIES
    preset.pipeline_model_overrides[model_kind] = {
        "num_geometry_supernodes": num_geometry_supernodes,
        "num_geometry_points": num_geometry_points,
        "num_surface_anchor_points": num_surface_anchor_points,
        "num_volume_anchor_points": num_volume_anchor_points,
    }

    from noether.core.schemas.modules.blocks import TransformerBlockConfig
    from noether.core.schemas.modules.encoders import SupernodePoolingConfig as SPConfig

    spool_cfg = SPConfig(hidden_dim=hidden_dim, input_dim=3, radius=supernode_radius)
    block_cfg = TransformerBlockConfig(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        mlp_expansion_factor=mlp_expansion_factor,
        use_rope=True,
        bias=True,
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
            name="abupt_autoencoder",
            hidden_dim=hidden_dim,
            supernode_pooling_config=spool_cfg,
            transformer_block_config=block_cfg,
            geometry_depth=geometry_depth,
            physics_blocks=physics_blocks,
            num_surface_blocks=num_surface_blocks,
            num_volume_blocks=num_volume_blocks,
            latent_dim=latent_dim,
            surface_field_dim=surface_field_dim,
            volume_field_dim=volume_field_dim,
            query_ratio=query_ratio,
            latent_num_surface_tokens=latent_num_surface_tokens,
            latent_num_volume_tokens=latent_num_volume_tokens,
            bottleneck_num_heads=bottleneck_num_heads,
            bottleneck_mode=bottleneck_mode,
        ),
        trainer_kind="noether.training.trainers.WeightedLossTrainer",
        trainer_params=dict(field_weights=DRIVAERML_FIELD_WEIGHTS, precision=precision),
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


def run_abupt_ae_pretrain(dataset_root: str, device: str = "cuda", **kwargs: Any) -> None:
    config = build_abupt_ae_pretrain_config(dataset_root=dataset_root, **kwargs)
    HydraRunner.main(device=device, config=config)
