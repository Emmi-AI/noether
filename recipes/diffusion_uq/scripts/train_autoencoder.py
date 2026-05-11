#!/usr/bin/env python3
#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Train AB-UPT autoencoder on DrivAerML.

Usage:
    python -m steady_diffusion.scripts.train_autoencoder \
        --dataset-root /path/to/drivaerml/preprocessed/subsampled_10x \
        --output-path ./outputs/abupt_ae \
        --max-epochs 100 --batch-size 8 --lr 5e-5 \
        --wandb-project noether-diffusion --wandb-entity myteam
"""

from __future__ import annotations

import argparse

import torch
from autoencoder_experiments import build_abupt_ae_pretrain_config

from noether.core.schemas.trackers import WandBTrackerSchema
from noether.training.runners import HydraRunner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AB-UPT autoencoder")

    # data
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--output-path", type=str, default="./outputs/abupt_ae")

    # architecture
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--latent-dim", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=3)
    p.add_argument("--geometry-depth", type=int, default=1)
    p.add_argument("--num-surface-blocks", type=int, default=6)
    p.add_argument("--num-volume-blocks", type=int, default=6)
    p.add_argument("--surface-field-dim", type=int, default=4)
    p.add_argument("--volume-field-dim", type=int, default=7)
    # mesh sampling
    p.add_argument("--num-geometry-supernodes", type=int, default=16384)
    p.add_argument("--num-geometry-points", type=int, default=65536)
    p.add_argument("--num-surface-anchor-points", type=int, default=1024)
    p.add_argument("--num-volume-anchor-points", type=int, default=1024)
    p.add_argument("--supernode-radius", type=float, default=0.25)
    p.add_argument(
        "--query-ratio",
        type=float,
        default=0.0,
        help="Fraction of anchors used as decode-only queries (0=standard AE, >0=position-independent)",
    )
    p.add_argument(
        "--latent-num-surface-tokens",
        type=int,
        default=None,
        help="K surface latent tokens (perceiver/sampled; derived for grid)",
    )
    p.add_argument(
        "--latent-num-volume-tokens",
        type=int,
        default=None,
        help="K volume latent tokens (perceiver/sampled; derived for grid)",
    )
    p.add_argument(
        "--bottleneck-num-heads", type=int, default=4, help="Attention heads for optional bottleneck cross-attn blocks"
    )
    p.add_argument(
        "--bottleneck-mode",
        type=str,
        default="perceiver",
        choices=[None, "perceiver", "sampled"],
        help="Token bottleneck mode. None = no bottleneck (legacy: "
        "auto-resolves to 'perceiver' if latent_num_*_tokens set).",
    )

    # training
    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup-percent", type=float, default=0.05, help="Fraction of training for linear LR warmup")
    p.add_argument("--end-lr", type=float, default=1e-6, help="Final LR for cosine decay (None to disable scheduling)")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument(
        "--precision", type=str, default="float32", choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"]
    )
    p.add_argument("--eval-every-n-epochs", type=int, default=1, help="Run test-set evaluation every N epochs")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    # wandb
    p.add_argument("--wandb-project", type=str, default="steady_diffusion_gg")
    p.add_argument("--wandb-entity", type=str, default="emmi-ai")
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument(
        "--wandb-tags",
        type=str,
        nargs="*",
        default=None,
        help="Tags for the W&B run (e.g. --wandb-tags sampled bottleneck)",
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    config = build_abupt_ae_pretrain_config(
        dataset_root=args.dataset_root,
        output_path=args.output_path,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_heads=args.num_heads,
        geometry_depth=args.geometry_depth,
        num_surface_blocks=args.num_surface_blocks,
        num_volume_blocks=args.num_volume_blocks,
        surface_field_dim=args.surface_field_dim,
        volume_field_dim=args.volume_field_dim,
        num_geometry_supernodes=args.num_geometry_supernodes,
        num_geometry_points=args.num_geometry_points,
        num_surface_anchor_points=args.num_surface_anchor_points,
        num_volume_anchor_points=args.num_volume_anchor_points,
        supernode_radius=args.supernode_radius,
        query_ratio=args.query_ratio,
        latent_num_surface_tokens=args.latent_num_surface_tokens,
        latent_num_volume_tokens=args.latent_num_volume_tokens,
        bottleneck_num_heads=args.bottleneck_num_heads,
        bottleneck_mode=args.bottleneck_mode,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_percent=args.warmup_percent,
        end_lr=args.end_lr if args.end_lr > 0 else None,
        weight_decay=args.weight_decay,
        clip_grad_norm=args.clip_grad_norm,
        num_workers=args.num_workers,
        precision=args.precision,
        eval_every_n_epochs=args.eval_every_n_epochs,
    )

    config.seed = args.seed
    import os

    from noether.core.providers.path import PathProvider

    run_id = PathProvider.generate_run_id()
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        run_id = f"{job_id}_{run_id}"
    config.run_id = run_id
    config.name = f"{config.model.name}_{run_id}"

    if args.wandb_project:
        config.tracker = WandBTrackerSchema(
            kind="noether.core.trackers.WandBTracker",
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            tags=args.wandb_tags,
        )

    print(f"device: {args.device}")
    if args.device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
        print(f"memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    HydraRunner.main(device=args.device, config=config)


if __name__ == "__main__":
    main()
