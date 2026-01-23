# ShapeNet-Car CFD Dataset

This dataset contains Computational Fluid Dynamics (CFD) simulation data for car models from the ShapeNet dataset. The dataset includes surface and volume properties for 889 different car geometries across 9 parameter configurations.

## Dataset Source

Original dataset: http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip

## Dataset Overview

- **Total Samples**: 889 simulations
- **Training Set**: param1-8 (789 samples)
- **Test/Validation Set**: param0 (100 samples)
- **Data Types**: Surface pressure, velocity fields, normals, and signed distance fields (SDF)

## Quick Start

### 1. Download Raw Data

Download and extract the dataset from the official source:

```bash

cd <path to store the downloaded data>
# Download dataset
wget http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip
unzip mlcfd_data.zip
rm mlcfd_data.zip 
rm -rf __MACOSX
cd mlcfd_data/training_data
# Extract all parameter folders
for i in {0..8}; do
    tar -xvf param${i}.tar.gz
done

# remove all tar files
for i in {0..8}; do
    rm param${i}.tar.gz
done
# remove folders without quadpress_smpl.vtk
rm -rf ./training_data/param2/854bb96a96a4d1b338acbabdc1252e2f
rm -rf ./training_data/param2/85bb9748c3836e566f81b21e2305c824
rm -rf ./training_data/param5/9ec13da6190ab1a3dd141480e2c154d3
rm -rf ./training_data/param8/c5079a5b8d59220bc3fb0d224baae2a

```

### 2. Preprocess the Data

#### Validate Preprocessing (Dry Run)

Before processing the entire dataset, validate that preprocessing works correctly:

```bash
python preprocessing.py \
    --root /path/to/mlcfd_data/training_data \
    --output_dir /path/to/output \
    --dry-run
```

The `--dry-run` flag will validate all data files without saving preprocessed outputs.

#### Run Full Preprocessing

Once validation succeeds, process the entire dataset:

```bash
python preprocessing.py \
    --root /path/to/mlcfd_data/training_data \
    --output_dir /path/to/output
```

**Output**: Preprocessed data will be stored at `/path/to/output/preprocessed/`

#### Advanced Preprocessing Options

Additional flags for robust preprocessing:

```bash
python preprocessing.py \
    --root /path/to/mlcfd_data/training_data \
    --output_dir /path/to/output \
    --continue-on-error \  # Continue if individual samples fail
    --verbose \             # Enable detailed logging
    --overwrite             # Allow overwriting existing output
```

**Available Options**:
- `--root`: Path to raw training data directory
- `--output_dir`: Directory for preprocessed output
- `--dry-run`: Validate without saving (useful for debugging)
- `--continue-on-error`: Continue processing even if some samples fail
- `--verbose, -v`: Enable DEBUG level logging
- `--overwrite`: Allow overwriting existing output directory

### 3. Verify Preprocessing

After preprocessing completes, verify the output structure:

```bash
tree -L 3 /path/to/output/preprocessed/
```

Expected structure:
```
preprocessed/
├── param0/
│   └── <simulation_id>/
│       ├── surface_points.pt
│       ├── surface_pressure.pt
│       ├── surface_normals.pt
│       ├── volume_points.pt
│       ├── volume_velocity.pt
│       ├── volume_normals.pt
│       └── volume_sdf.pt
├── param1/
│   └── ...
...
└── param8/
```

## Data Format

Each preprocessed simulation contains the following files:

| File | Shape | Description |
|------|-------|-------------|
| `surface_points.pt` | `(N_surf, 3)` | 3D coordinates of surface points |
| `surface_pressure.pt` | `(N_surf,)` | Pressure values at surface points |
| `surface_normals.pt` | `(N_surf, 3)` | Unit normal vectors at surface points |
| `volume_points.pt` | `(N_vol, 3)` | 3D coordinates of volume points |
| `volume_velocity.pt` | `(N_vol, 3)` | Velocity vectors at volume points |
| `volume_normals.pt` | `(N_vol, 3)` | Unit normal vectors at volume points |
| `volume_sdf.pt` | `(N_vol,)` | Signed distance field w.r.t. car body |

## Computing the statistics of the dataset

```bash 
noether-dataset-stats \
    --dataset_kind=noether.data.datasets.cfd.shapenet_car.ShapeNetCarDataset \
    --split=train \
    --root=<path to dataset root> \
    --exclude_attributes=surface_friction,volume_pressure,volume_vorticity
```