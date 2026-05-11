# Diffusion UQ on DrivAerML

Generative uncertainty quantification for CFD fields on top of the AB-UPT
backbone via **data-space diffusion** — denoise surface/volume fields directly.

See [`REPORT.md`](./REPORT.md) for architecture and sampling details.

## Quick start

```bash
# allocate a GPU node
salloc --cpus-per-task=28 --mem=250GB --reservation=dev --gpus-per-node=1 --time 1-0 srun --pty zsh

cd ~/exp/noether && source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/diffusion_uq:$PYTHONPATH"

DATASET=/nfs-gpu/research/datasets/drivaerml/preprocessed/subsampled_10x
```


### Data-space diffusion

```bash
python -m steady_diffusion.scripts.train_dataspace_diffusion \
    --dataset-root $DATASET \
    --output-path ./outputs/abupt_diffusion \
    --paradigm flow_matching \
    --max-epochs 500 --batch-size 1 --lr 5e-5
```

## SLURM

| File | Stage |
|---|---|
| `run_abupt_diffusion.sbatch` | data-space diffusion |

The launcher prepends `$SLURM_JOB_ID` to `config.run_id` so wandb + output
dirs trace back to the job.

## Diffusion paradigms

| Paradigm | Loss | Sampler | Default steps |
|---|---|---|---|
| `flow_matching` | Velocity (rectified flow) | Euler | 50 |

## Notebooks

- `01_noether_aero_cfd_guide.ipynb` — noether + DrivAerML tutorial.
- `03_dataspace_diffusion.ipynb` — data-space full-mesh chunked eval.

## Project structure

```
diffusion_uq/
├── REPORT.md
├── dataspace.md                  # data-space ablations
├── steady_diffusion/
│   ├── experiments.py            # config factories
│   ├── models/
│   │   └── diffusion_abupt.py    # DiffusionABUPT (dataspace)
│   ├── schemas/                  # pydantic configs
│   ├── diffusion/                # FlowMatching
│   ├── datasets/                 # DrivAerML pipeline
│   ├── trainer/
│   ├── callbacks/
│   ├── scripts/
│   │   └── train_dataspace_diffusion.py
│   └── viz.py
```
