#!/bin/bash
#SBATCH --job-name=latent_diffusion
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --gpus-per-node=1
#SBATCH --time=2-0
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd /home/ggalletti/exp/noether
export PYTHONPATH=/home/ggalletti/exp/noether/diffusion_uq:${PYTHONPATH:-}

# point at the AE checkpoint to use; latent_root + output_path are auto-derived
# from the AE_TAG (trailing token of the run dir).
AE_CHECKPOINT="/home/ggalletti/exp/noether/outputs/abupt_ae_bottleneck/18873_2026-04-18_ada5a/checkpoints/abupt_autoencoder_cp=best_model.loss.test.total_model.th"

python -m steady_diffusion.scripts.train_latent_diffusion \
    --ae-checkpoint "$AE_CHECKPOINT" \
    --dataset-root /nfs-gpu/research/datasets/drivaerml/preprocessed/subsampled_10x \
    --paradigm flow_matching \
    --hidden-dim 384 \
    --denoiser-depth 4 \
    --denoiser-heads 6 \
    --max-epochs 500 \
    --batch-size 32 \
    --lr 4e-3 \
    --weight-decay 0.05 \
    --clip-grad-norm 1.0 \
    --num-workers 2 \
    --precision float32 \
    --wandb-project steady_diffusion_gg \
    --wandb-entity emmi-ai \
    --wandb-mode online
