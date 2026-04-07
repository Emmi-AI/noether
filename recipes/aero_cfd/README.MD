# Aerodynamic CFD Recipe

**Authors**: Maurits Bleeker, Markus Hennerbichler, and Pavel Kuksa

This recipe demonstrates how to use the [Noether Framework](https://github.com/Emmi-AI/noether) for aerodynamics Computational Fluid Dynamics (CFD) tasks, based on the experiments from Section 4.4 of the [AB-UPT paper](https://arxiv.org/pdf/2502.09692).

## Overview

The recipe provides a complete training pipeline for predicting aerodynamic quantities (pressure, velocity, etc.) on surface and volume meshes of car geometries. It includes:

- **Datasets**: Configurations for ShapeNet-Car, AhmedML, DrivAerML, DrivAerNet++, and Emmi-Wing
- **Models**: Wrappers for Transformer, Transolver, Transolver++, UPT, and AB-UPT architectures
- **Pipeline**: A multi-stage data pipeline handling point sampling, collation, and target assignment for both point-based and query-based models
- **Trainer**: A weighted multi-objective loss trainer supporting surface and volume predictions
- **Callbacks**: Evaluation metrics (MSE, MAE, Relative L2) with denormalization and chunked inference
- **Configs**: Hydra + Pydantic experiment configurations with SLURM job scripts

There are two ways to run experiments with this recipe: using **YAML configs** (via the CLI) or using **Python run scripts** (via the preset-based interface).

## Option 1: Run with configs

```bash
# From the recipe folder (recipes/aero_cfd/)
uv run noether-train \
  --hp configs/train_shapenet.yaml \
  +experiment/shapenet=transformer tracker=disabled +seed=1
```

> [!IMPORTANT]
> All CLI commands in this recipe must be run from the `recipes/aero_cfd/` directory.
> Set `dataset_root` in the config file or via CLI override before running.

## Option 2: Run with Python scripts

The `scripts/` folder contains Python training scripts that define the same experiments using the preset-based interface — no YAML files needed.

| Script | Dataset | Source |
|---|---|---|
| `scripts/train_ahmedml.py` | AhmedML (CAEML benchmark) | [CAEML](https://caeml.org/) |
| `scripts/train_drivaerml.py` | DrivAerML (CAEML benchmark) | [CAEML](https://caeml.org/) |
| `scripts/train_drivaernet.py` | DrivAerNet++ | [DrivAerNet](https://github.com/Mohamedelrefaie/DrivAerNet) |
| `scripts/train_emmi_wing.py` | Emmi Wing | Internal |
| `scripts/train_shapenet_car.py` | ShapeNet Car | [ShapeNet](https://shapenet.org/) |

Each script supports the following model architectures: AB-UPT, UPT, Transformer, and Transolver.

From the `recipes` directory run:

```console
uv run python -m aero_cfd.scripts.train_shapenet_car \
  --dataset-root /path/to/shapenet_car \
  --output-path /path/to/outputs \
  --accelerator gpu \
  --model abupt
```

Available arguments:

| Argument | Description |
|---|---|
| `--dataset-root` | **(required)** Path to the dataset. |
| `--output-path` | **(required)** Path to store training outputs. |
| `--accelerator` | Accelerator to use: `cpu`, `gpu`, or `mps` (default: `gpu`). |
| `--model` | Model architecture: `abupt`, `upt`, `transformer`, or `transolver` (default: `abupt`). |

## Documentation

For a detailed walkthrough of this recipe covering configuration, datasets, pipeline, models, callbacks, and experiment execution, see the [full tutorial in the documentation](https://noether-docs.emmi.ai/html/tutorials/walkthrough/index.html).

## Project structure

```
recipes/aero_cfd/
├── callbacks/        # Evaluation and metrics callbacks
├── configs/          # Hydra YAML configurations
│   ├── callbacks/
│   ├── data_specs/
│   ├── dataset_normalizers/
│   ├── datasets/
│   ├── experiment/
│   ├── model/
│   ├── optimizer/
│   ├── pipeline/
│   ├── tracker/
│   └── trainer/
├── jobs/             # SLURM job scripts
├── model/            # Model architecture wrappers
├── pipeline/         # Multi-stage data pipeline
├── schemas/          # Pydantic configuration schemas
├── scripts/          # Python training scripts (preset-based)
│   ├── presets/      # Dataset presets for Python interface
│   └── train_*.py    # Per-dataset run scripts
└── trainers/         # Custom trainer implementation
```

## Questions

For questions, please create an issue in our [GitHub repo](https://github.com/Emmi-AI/noether/issues).
