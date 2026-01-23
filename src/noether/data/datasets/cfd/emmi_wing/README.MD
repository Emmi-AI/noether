# Emmi Wing Dataset

This dataset contains CFD simulation data for Wings.

## Dataset Source

Produced by Emmi AI.

## Dataset Overview

Detailed information can be found on [HuggingFace](https://huggingface.co/datasets/EmmiAI/Emmi-Wing) or 
[GitHub](https://github.com/Emmi-AI/Emmi-Wing) and in the paper [Going with the Speed of Sound: Pushing Neural Surrogates into Highly-turbulent Transonic Regimes](https://arxiv.org/abs/2511.21474).

## Quick Start

### 1. Download Raw Data

https://data.emmi.ai/s/qTgKFQCRNnFTgXN

### 2. Preprocess the Data

The data already provides the tensors that can be used for simulation.

## Data Format

```
run_X.zip
├── design_parameters.pt
├── surface_position.pt
├── surface_pressure.pt 
├── surface_wall_shear_stress.pt 
├── volume_position.pt
├── volume_pressure.pt
├── volume_velocity.pt
├── volume_vorticity.pt
├── cell_areas.npy
├── cell_centers.npy
├── cell_normals.npy
└── wing.stl
```

## Computing the Statistics of the Dataset

```bash
noether-dataset-stats \
    --dataset_kind=noether.data.datasets.cfd.emmi_wing.EmmiWingDataset \
    --split=train \
    --exclude_attributes=surface_normals,volume_normals,volume_sdf \
    --log_scale=volume_vorticity \
    --root=<root to dataset>
```

