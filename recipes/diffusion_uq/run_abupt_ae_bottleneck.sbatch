#!/bin/bash
#SBATCH --job-name=abupt_ae_bn
#SBATCH --cpus-per-task=28
#SBATCH --mem=250GB
#SBATCH --gpus-per-node=1
#SBATCH --time=2-0
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd /home/ggalletti/exp/noether
export PYTHONPATH=/home/ggalletti/exp/noether/diffusion_uq:${PYTHONPATH:-}

python -m steady_diffusion.scripts.train_autoencoder \
    --dataset-root /nfs-gpu/research/datasets/drivaerml/preprocessed/subsampled_10x \
    --output-path ./outputs/abupt_ae_bottleneck \
    --num-geometry-points 65536 \
    --num-geometry-supernodes 16384 \
    --num-surface-anchor-points 32768 \
    --num-volume-anchor-points 32768 \
    --latent-num-surface-tokens 8192 \
    --latent-num-volume-tokens 8192 \
    --bottleneck-num-heads 4 \
    --bottleneck-mode sampled \
    --latent-dim 128 \
    --query-ratio 0.5 \
    --eval-every-n-epochs 5 \
    --max-epochs 500 \
    --batch-size 8 \
    --num-workers 16 \
    --precision bfloat16 \
    --lr 1e-4 \
    --weight-decay 0.1 \
    --clip-grad-norm 1.0 \
    --wandb-project steady_diffusion_gg \
    --wandb-entity emmi-ai \
    --wandb-mode online
