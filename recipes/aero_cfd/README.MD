# Aerodynamic CFD Recipe

**Authors**: Maurits Bleeker, Markus Hennerbichler, and Pavel Kuksa

This recipe demonstrates how to use the [Noether Framework](https://github.com/Emmi-AI/noether) for automotive Computational Fluid Dynamics (CFD) tasks, based on the experiments from Section 4.4 of the [AB-UPT paper](https://arxiv.org/pdf/2502.09692).

## Overview

The recipe provides a complete training pipeline for predicting aerodynamic quantities (pressure, velocity, etc.) on surface and volume meshes of car geometries. It includes:

- **Datasets**: Configurations for ShapeNet-Car, AhmedML, DrivAerML, DrivAerNet++, and Emmi-Wing
- **Models**: Wrappers for Transformer, Transolver, Transolver++, UPT, and AB-UPT architectures
- **Pipeline**: A multi-stage data pipeline handling point sampling, collation, and target assignment for both point-based and query-based models
- **Trainer**: A weighted multi-objective loss trainer supporting surface and volume predictions
- **Callbacks**: Evaluation metrics (MSE, MAE, Relative L2) with denormalization and chunked inference
- **Configs**: Hydra + Pydantic experiment configurations with SLURM job scripts

## Quick start

```bash
# From the recipe folder (recipes/aero_cfd/)
uv run noether-train \
  --hp configs/train_shapenet.yaml \
  +experiment/shapenet=transformer tracker=disabled +seed=1
```

> [!IMPORTANT]
> All commands in this recipe must be run from the `recipes/aero_cfd/` directory.
> Set `dataset_root` in the config file or via CLI override before running.

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
└── trainers/         # Custom trainer implementation
```

## Questions

For questions, please create an issue in our [GitHub repo](https://github.com/Emmi-AI/noether/issues).
