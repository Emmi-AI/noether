# Rotary Normal Embedding (RoNE) Example

This example demonstrates how to extend the AB-UPT with our preliminary work in **Rotary Normal Embedding (RoNE)** for improved geometric encoding.

## Overview

RoNE extends standard Rotary Position Embedding (RoPE) by incorporating surface normal vectors alongside positions, enabling the model to encode both spatial location and surface orientation. 

## Example Structure

```
rotary_normal_embedding/
├── model.py                          # Extended AB-UPT from the tutorial with RoNE support
├── pipeline.py                       # Extended the data pipeline from the tutorial with normal handling
├── schemas.py                        # Configuration schema for RoNE model. Mainly extends the schema's from the tutorial
├── README.md                      
├── configs/                          # Configuration files
│   ├── train_shapenet.yaml          # Main config for ShapeNet-Car dataset
│   ├── train_drivaerml.yaml         # Main config for DrivAerML dataset
│   ├── model/
│   │   └── ab_upt.yaml              # Model architecture configuration
│   ├── data_specs/                  # Dataset specifications
│   │   ├── shapenet_car.yaml        # ShapeNet field dimensions
│   │   └── caeml.yaml               # CAEML (AhmedML/DrivAerML) specs
│   ├── pipeline/                    # Data pipeline configs
│   ├── trainer/                     # Training configs
│   ├── callbacks/                   # Callback configs
│   └── experiment/                  # Experiment-specific configs
│       ├── shapenet/ab_upt.yaml
│       └── drivaerml/ab_upt.yaml
├── datasets/
│   └── caeml/
│       └── dataset.py               # Override of the CAEML with a different file map
└── jobs/                            # SLURM job scripts
    ├── train_normal_rope_shapenet.job
    ├── train_normal_rope_drivaerml.job
    └── experiments/                 # Experiment configurations
        ├── normal_rope_shapenet.txt
        └── normal_rope_drivaerml.txt
```

### Configuration Files

4. **`configs/model/ab_upt.yaml`**
   - Key RoNE settings:
     ```yaml
     use_normal_rope: true
     use_surface_normal_features: true
     normal_rope_dim_fraction: 0.15
     normal_rope_max_wavelength: 10.0
     cross_attention_normal_mode: position_only
     supernode_pooling_config:
       input_features_dim: 3  # Use normals as input features
     ```

## Running Experiments

In the jobs folder, we provide the SLURM job files and job-array to run a baseline AB-UPT and a AB-UPT with RoNE on both DrivAerMl and ShapeNet-Car.
In the following [WandB report](https://wandb.ai/emmi-ai/geometry-encoding-validation/reports/Rotary-Normal-Embedding-RoNE---VmlldzoxNjA0NjIwNw
) we provide some of the results of those experiments: 

## License

Copyright © 2025 Emmi AI GmbH. All rights reserved.
