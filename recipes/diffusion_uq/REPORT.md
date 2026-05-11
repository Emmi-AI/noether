# Uncertainty Quantification for CFD via Diffusion on AB-UPT

## Context

Deterministic AB-UPT regression produces a single field prediction per geometry — it
has no notion of epistemic or aleatoric uncertainty. This project adds **generative
modeling** on top of the AB-UPT backbone so that every inference produces a
distribution of plausible fields. Sampling an ensemble gives per-point mean/std,
calibration of integrated quantities (drag, lift), and high-frequency detail that
the deterministic regression smooths out.

See [`dataspace.md`](./dataspace.md) for the data-space ablations.

## Data-space diffusion

`DiffusionABUPT` wraps the AB-UPT backbone (geometry encoder → physics blocks →
surface/volume decoders) and denoises surface+volume fields directly at anchor
positions. The only data-space-specific additions on top of the regression
backbone are noisy-field projection into anchor embeddings and DiT timestep
modulation on every block. One forward pass per sampling step. See
[`dataspace.md`](./dataspace.md) for the ablation table (optimizers,
schedules). Flow matching with 50 Euler steps is the chosen default.

## Training recipes

| Stage | sbatch | Script |
|---|---|---|
| Data-space diffusion | `run_abupt_diffusion.sbatch` | `scripts/train_dataspace_diffusion.py` |

SLURM job id is prepended to `config.run_id` so wandb + output dirs trace back
to the job.

## Evaluation

- **`03_dataspace_diffusion.ipynb`** — full-resolution (50K + 50K) chunked
  eval for the data-space model.

## Caveats

- **Checkpoints from before the model rename will not auto-resume.** Saved
  `hp_resolved.yaml` snapshots record old `kind=` paths (e.g.
  `steady_diffusion.models.diffusion_ab_upt.DiffusionABUPT`) resolved at load
  time. Raw `.th` state-dict files are safe (keyed by parameter names); rebuild
  the config from scratch and load the state dict manually.
