#!/usr/bin/env python3
#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Train latent diffusion (flow matching) on top of a pretrained AB-UPT autoencoder.

Mirrors notebook 02_steady_diffusion.ipynb cell-17 — derives AE arch from
the hp_resolved.yaml stored next to the checkpoint, derives use_rope /
num_surface_tokens from the AE bottleneck mode, and trains a small DiT
denoiser in latent space.

Usage:
    python -m steady_diffusion.scripts.train_latent_diffusion \\
        --ae-checkpoint /path/to/abupt_autoencoder_cp=...model.th \\
        --dataset-root /nfs-gpu/research/datasets/drivaerml/preprocessed/subsampled_10x
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml
from autoencoder_experiments import build_abupt_ae_pretrain_config
from latent_diffusion_experiments import build_latent_diffusion_config
from scripts.extract_latents import extract_latents

from noether.core.providers.path import PathProvider
from noether.core.schemas.trackers import WandBTrackerSchema
from noether.training.runners import HydraRunner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train latent diffusion on AB-UPT AE")

    # data / io
    p.add_argument(
        "--ae-checkpoint",
        type=str,
        required=True,
        help="Path to .th AE checkpoint. hp_resolved.yaml must sit two dirs up.",
    )
    p.add_argument(
        "--dataset-root", type=str, default="/nfs-gpu/research/datasets/drivaerml/preprocessed/subsampled_10x"
    )
    p.add_argument(
        "--latent-root",
        type=str,
        default=None,
        help="Where to read/write extracted latents. Default: ~/exp/noether/data/latents_abupt_<AE_TAG>",
    )
    p.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Diffusion run output dir. Default: ~/exp/noether/outputs/latent_diffusion_fm_abupt_<AE_TAG>",
    )
    p.add_argument(
        "--extract-batch-size",
        type=int,
        default=8,
        help="Batch size for latent extraction (only if not already cached).",
    )

    # diffusion
    p.add_argument("--paradigm", type=str, default="flow_matching", choices=["flow_matching"])

    # denoiser arch (notebook cell-17 defaults)
    p.add_argument("--hidden-dim", type=int, default=384)
    p.add_argument("--denoiser-depth", type=int, default=4)
    p.add_argument("--denoiser-heads", type=int, default=6)
    p.add_argument(
        "--condition-dim",
        type=int,
        default=None,
        help="Time-condition dim. Defaults to AE latent_dim (notebook behavior).",
    )
    p.add_argument(
        "--adaln-zero-std", type=float, default=None, help="Init std for adaLN modulation linear. None = default init."
    )

    # training (notebook cell-17 defaults)
    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=4e-3)
    p.add_argument("--warmup-percent", type=float, default=0.05)
    p.add_argument("--end-lr", type=float, default=1e-6, help="Final LR for cosine decay (0 to disable scheduling)")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--precision", type=str, default="float32", choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"]
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-save-checkpoints", action="store_true", help="Disable checkpoint callbacks (default: save).")

    # resume / warm-start
    p.add_argument("--resume-run-id", type=str, default=None)
    p.add_argument("--resume-checkpoint", type=str, default="best_model.loss.test.total")
    p.add_argument("--warm-start", action="store_true")

    # wandb
    p.add_argument("--wandb-project", type=str, default="steady_diffusion_gg")
    p.add_argument("--wandb-entity", type=str, default="emmi-ai")
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-tags", type=str, nargs="*", default=None)

    return p.parse_args(argv)


def _build_ae_config(ckpt_path: Path, dataset_root: str, output_path: str):
    """Mirror notebook cell-6: parse hp_resolved.yaml, rebuild AE config."""
    run_dir = ckpt_path.parent.parent
    hp_path = run_dir / "hp_resolved.yaml"
    assert hp_path.exists(), f"hp_resolved.yaml not found at {hp_path}"

    with open(hp_path) as f:
        hp = yaml.full_load(f)

    m = hp["model"]
    pl = hp["datasets"]["train"]["pipeline"]

    ae_config = build_abupt_ae_pretrain_config(
        dataset_root=dataset_root,
        output_path=output_path,
        hidden_dim=m["hidden_dim"],
        latent_dim=m["latent_dim"],
        num_heads=m["transformer_block_config"]["num_heads"],
        geometry_depth=m["geometry_depth"],
        num_surface_blocks=m["num_surface_blocks"],
        num_volume_blocks=m["num_volume_blocks"],
        surface_field_dim=m["surface_field_dim"],
        volume_field_dim=m["volume_field_dim"],
        num_geometry_supernodes=pl["num_geometry_supernodes"],
        num_geometry_points=pl["num_geometry_points"],
        num_surface_anchor_points=pl["num_surface_anchor_points"],
        num_volume_anchor_points=pl["num_volume_anchor_points"],
        supernode_radius=m["supernode_pooling_config"].get("radius", 0.25),
        query_ratio=m.get("query_ratio", 0.0),
        latent_num_surface_tokens=m.get("latent_num_surface_tokens"),
        latent_num_volume_tokens=m.get("latent_num_volume_tokens"),
        bottleneck_num_heads=m.get("bottleneck_num_heads", 4),
        bottleneck_mode=m.get("bottleneck_mode"),
        max_epochs=1,
        batch_size=1,
    )
    return ae_config, m, pl


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    ckpt_path = Path(args.ae_checkpoint)
    assert ckpt_path.exists(), f"AE checkpoint not found: {ckpt_path}"

    # AE_TAG = trailing token of run dir (e.g. "ada5a" from "18873_2026-04-18_ada5a")
    ae_tag = ckpt_path.parent.parent.name.split("_")[-1]
    home = os.path.expanduser("~")
    latent_root = args.latent_root or f"{home}/exp/noether/data/latents_abupt_{ae_tag}"
    output_path = args.output_path or f"{home}/exp/noether/outputs/latent_diffusion_fm_abupt_{ae_tag}"
    print(f"AE_TAG={ae_tag}")
    print(f"latent_root={latent_root}")
    print(f"output_path={output_path}")

    # rebuild AE config from hp_resolved.yaml
    ae_run_root = str(ckpt_path.parent.parent.parent)
    ae_config, m, pl = _build_ae_config(ckpt_path, args.dataset_root, ae_run_root)

    # extract latents if missing (mirrors cell-8)
    if (Path(latent_root) / "train_stats.pt").exists():
        print(f"latents already extracted at {latent_root}")
    else:
        print(f"extracting latents → {latent_root}")
        extract_latents(
            ae_config,
            output_root=latent_root,
            device=args.device,
            batch_size=args.extract_batch_size,
            checkpoint_path=str(ckpt_path),
        )

    stats = torch.load(f"{latent_root}/train_stats.pt", weights_only=True)
    latent_scale = float(stats["latent_scale"])
    print(f"latent_scale={latent_scale:.4f}")

    # derive denoiser arch knobs from AE bottleneck (mirrors cell-17)
    K_s = m.get("latent_num_surface_tokens")
    K_v = m.get("latent_num_volume_tokens")
    bmode = m.get("bottleneck_mode")
    has_bottleneck = K_s is not None and K_v is not None
    if has_bottleneck:
        n_surf_tokens = K_s
        use_rope = bmode == "sampled"
    else:
        n_surf_tokens = pl.get("num_surface_anchor_points", 1024)
        use_rope = True

    condition_dim = args.condition_dim if args.condition_dim is not None else m["latent_dim"]

    config = build_latent_diffusion_config(
        latent_root=latent_root,
        cfd_root=args.dataset_root,
        ae_config=ae_config,
        output_path=output_path,
        paradigm=args.paradigm,
        latent_scale=latent_scale,
        latent_dim=m["latent_dim"],
        hidden_dim=args.hidden_dim,
        denoiser_depth=args.denoiser_depth,
        denoiser_heads=args.denoiser_heads,
        denoiser_use_rope=use_rope,
        num_surface_tokens=n_surf_tokens,
        condition_dim=condition_dim,
        adaln_zero_std=args.adaln_zero_std,
        num_geometry_supernodes=pl["num_geometry_supernodes"],
        num_geometry_points=pl["num_geometry_points"],
        num_surface_anchor_points=pl["num_surface_anchor_points"],
        num_volume_anchor_points=pl["num_volume_anchor_points"],
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_percent=args.warmup_percent,
        end_lr=args.end_lr if args.end_lr > 0 else None,
        weight_decay=args.weight_decay,
        clip_grad_norm=args.clip_grad_norm,
        precision=args.precision,
        save_checkpoints=not args.no_save_checkpoints,
        num_workers=args.num_workers,
    )

    config.seed = args.seed
    # prepend SLURM job id to run_id so wandb + output dirs trace back to the job
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
    print(f"paradigm: {args.paradigm}, run_id: {run_id}")
    print(
        f"denoiser: hidden={args.hidden_dim} depth={args.denoiser_depth} "
        f"heads={args.denoiser_heads} rope={use_rope} n_surf_tokens={n_surf_tokens}"
    )

    if args.resume_run_id:
        config.resume_run_id = args.resume_run_id
        config.resume_checkpoint = args.resume_checkpoint

        if args.warm_start:
            from noether.core.schemas.initializers import PreviousRunInitializerConfig

            init_cls = PreviousRunInitializerConfig
            print(f"warm-start from {args.resume_run_id} @ {args.resume_checkpoint}")
        else:
            from noether.core.schemas.initializers import ResumeInitializerConfig

            init_cls = ResumeInitializerConfig
            print(f"resuming from {args.resume_run_id} @ {args.resume_checkpoint}")

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
