#!/bin/bash
#SBATCH --job-name=abupt_ae
#SBATCH --cpus-per-task=28
#SBATCH --mem=250GB
#SBATCH --gpus-per-node=1
#SBATCH --time=2-0
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd /home/ggalletti/exp/noether
export PYTHONPATH=/home/ggalletti/exp/noether/diffusion_uq:${PYTHONPATH:-}

# Position-independent ABUPT autoencoder: encode at 50% of anchors,
# decode at all anchors. Forces the latent to be a general field encoding
# not tied to specific positions. 2x anchor points to keep encode set ~1K.
python -m steady_diffusion.scripts.train_autoencoder \
    --dataset-root /nfs-gpu/research/datasets/drivaerml/preprocessed/subsampled_10x \
    --output-path ./outputs/abupt_ae_query \
    --num-geometry-points 65536 \
    --num-geometry-supernodes 16384 \
    --num-surface-anchor-points 16384 \
    --num-volume-anchor-points 16384 \
    --query-ratio 0.5 \
    --eval-every-n-epochs 5 \
    --max-epochs 500 \
    --batch-size 8 \
    --num-workers 16 \
    --precision bfloat16 \
    --lr 4e-5 \
    --weight-decay 1e-5 \
    --clip-grad-norm 1.0 \
    --wandb-project steady_diffusion_gg \
    --wandb-entity emmi-ai \
    --wandb-mode online
