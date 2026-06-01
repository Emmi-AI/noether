#!/bin/bash
#SBATCH --job-name=abupt_diffusion
#SBATCH --cpus-per-task=28
#SBATCH --mem=256GB
#SBATCH --gpus-per-node=1
#SBATCH --time=2-0
#SBATCH --output=/home/%u/logs/diffusion/%j.out

set -euo pipefail

uv run python -m scripts.train_dataspace_diffusion \
    --dataset-root /nfs-gpu/research/datasets/drivaerml/preprocessed/subsampled_10x \
    --output-path ./outputs/abupt_diffusion \
    --hidden-dim 192 \
    --num-heads 3 \
    --geometry-depth 2 \
    --num-surface-blocks 6 \
    --num-volume-blocks 6 \
    --time-embed-dim 192 \
    --num-geometry-points 65536 \
    --num-geometry-supernodes 16384 \
    --num-surface-anchor-points 16384 \
    --num-volume-anchor-points 16384 \
    --max-epochs 500 \
    --batch-size 8 \
    --eval-every-n-epochs 5 \
    --lr 5e-5 \
    --weight-decay 0.05 \
    --clip-grad-norm 1.0 \
    --ema-decays 0.9999 \
    --ema-save-every-n-epochs 10 \
    --wandb-project steady_diffusion_gg \
    --wandb-entity emmi-ai \
    --wandb-mode online \
    --paradigm regression
