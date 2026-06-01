#!/usr/bin/env python3
#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Train data-space AB-UPT diffusion on DrivAerML (no AE, denoises fields directly).

Usage:
    python -m steady_diffusion.scripts.train_dataspace_diffusion \
        --dataset-root /path/to/drivaerml/preprocessed/subsampled_10x \
        --output-path ./outputs/diffusion_ab_upt \
        --paradigm flow_matching \
        --max-epochs 500 --batch-size 1 --lr 5e-5 \
        --wandb-project steady_diffusion_gg --wandb-entity emmi-ai
"""

from __future__ import annotations

import argparse

import torch
from experiments import build_abupt_config

from noether.core.schemas.trackers import WandBTrackerSchema
from noether.training.runners import HydraRunner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train data-space AB-UPT diffusion")

    # data / io
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--output-path", type=str, default="./outputs/diffusion_ab_upt")

    # architecture (defaults match dataspace.md, 9.1M params)
    p.add_argument("--paradigm", type=str, default="flow_matching", choices=["flow_matching", "regression"])
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--num-heads", type=int, default=3)
    p.add_argument("--mlp-expansion-factor", type=int, default=4)
    p.add_argument("--geometry-depth", type=int, default=1)
    p.add_argument("--num-surface-blocks", type=int, default=6)
    p.add_argument("--num-volume-blocks", type=int, default=6)
    p.add_argument("--time-embed-dim", type=int, default=256)
    p.add_argument(
        "--minibatch-ot",
        action="store_true",
        help="Whether to apply minibatch optimal transport (Pooladian et al. 2023)",
    )

    # mesh sampling
    p.add_argument("--num-geometry-supernodes", type=int, default=16384)
    p.add_argument("--num-geometry-points", type=int, default=65536)
    p.add_argument("--num-surface-anchor-points", type=int, default=16384)
    p.add_argument("--num-volume-anchor-points", type=int, default=16384)
    p.add_argument("--supernode-radius", type=float, default=0.25)

    # training
    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument(
        "--max-updates",
        type=int,
        default=None,
        help="Stop after N updates instead of N epochs (overrides --max-epochs). Useful for smoke tests.",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup-percent", type=float, default=0.05)
    p.add_argument("--end-lr", type=float, default=1e-6, help="Final LR for cosine decay (0 to disable scheduling)")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)
    p.add_argument(
        "--precision", type=str, default="float16", choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"]
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--eval-every-n-epochs",
        type=int,
        default=5,
        help="Periodic FM-sampling eval frequency (DataspaceDiffusionEvalCallback)",
    )

    # ema (weight exponential moving average — stabilizes late-stage diffusion)
    p.add_argument(
        "--ema-decays",
        type=float,
        nargs="*",
        default=None,
        help="EMA decay factors (e.g. --ema-decays 0.999 0.9999). Omit to disable EMA.",
    )
    p.add_argument("--ema-save-every-n-epochs", type=int, default=10, help="How often to checkpoint EMA weights")

    # resume / warm-start
    p.add_argument(
        "--resume-run-id", type=str, default=None, help="Run ID to load weights from (e.g. 18455_2026-04-15_fa31n)"
    )
    p.add_argument(
        "--resume-checkpoint",
        type=str,
        default="best_model.loss.test.total",
        help="Checkpoint tag (default: best model). Examples: latest, E100_U2500_S40000",
    )
    p.add_argument(
        "--warm-start", action="store_true", help="Load model weights only (fresh optimizer + LR schedule, epoch 0)"
    )

    # wandb
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default="emmi-ai")
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument(
        "--wandb-tags",
        type=str,
        nargs="*",
        default=None,
        help="Tags for the W&B run (e.g. --wandb-tags sampled bottleneck)",
    )
    p.add_argument(
        "--experiment-name", type=str, default=None, help="Optional name for the experiment (overrides default naming)"
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    config = build_abupt_config(
        dataset_root=args.dataset_root,
        paradigm=args.paradigm,
        output_path=args.output_path,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        mlp_expansion_factor=args.mlp_expansion_factor,
        geometry_depth=args.geometry_depth,
        num_surface_blocks=args.num_surface_blocks,
        num_volume_blocks=args.num_volume_blocks,
        time_embed_dim=args.time_embed_dim,
        num_geometry_supernodes=args.num_geometry_supernodes,
        num_geometry_points=args.num_geometry_points,
        num_surface_anchor_points=args.num_surface_anchor_points,
        num_volume_anchor_points=args.num_volume_anchor_points,
        supernode_radius=args.supernode_radius,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_percent=args.warmup_percent,
        end_lr=args.end_lr if args.end_lr > 0 else None,
        weight_decay=args.weight_decay,
        clip_grad_norm=args.clip_grad_norm,
        precision=args.precision,
        minibatch_ot=args.minibatch_ot,
        eval_every_n_epochs=args.eval_every_n_epochs,
        ema_decays=args.ema_decays,
        ema_save_every_n_epochs=args.ema_save_every_n_epochs,
    )

    if args.max_updates is not None:
        config.trainer.max_epochs = None
        config.trainer.max_updates = args.max_updates

    config.seed = args.seed
    # prepend SLURM job id (if set) so wandb/output dirs trace back to the job
    import os

    from noether.core.providers.path import PathProvider

    run_id = PathProvider.generate_run_id()
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        run_id = f"{job_id}_{run_id}"
    config.run_id = run_id
    config.name = args.experiment_name or f"{config.model.name}_{run_id}"

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
    print(f"run_id: {run_id}")

    if args.resume_run_id:
        config.resume_run_id = args.resume_run_id
        config.resume_checkpoint = args.resume_checkpoint

        if args.warm_start:
            from noether.core.schemas.initializers import PreviousRunInitializerConfig

            init_cls = PreviousRunInitializerConfig
            print(f"warm-start from {args.resume_run_id} @ {args.resume_checkpoint} (fresh optimizer)")
        else:
            from noether.core.schemas.initializers import ResumeInitializerConfig

            init_cls = ResumeInitializerConfig
            print(f"resuming from {args.resume_run_id} @ {args.resume_checkpoint} (full state)")

        trainer, model, tracker, mc = HydraRunner.setup_experiment(
            device=args.device,
            config=config,
            initializer_config_class=init_cls,
        )
        trainer.train(model)
        tracker.summarize_logvalues()
        mc.log()
        tracker.close()
    else:
        HydraRunner.main(device=args.device, config=config)


if __name__ == "__main__":
    main()
