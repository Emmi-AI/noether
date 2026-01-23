# AhmedML and DrivAerML CFD Dataset

This dataset contains CFD simulation data for from the AhmedML and DrivAerML datasets. 
The dataset includes surface and volume properties for 500 different car geometries.

## Dataset Source

Original datasets: https://caemldatasets.org/

## Dataset Overview

- **Total Samples**: 500 simulations
- **Training Set**: 400 samples
- **Test Set**: 50 samples
- **Validation Set**: 34 samples (for DrivAerML)
- **Hidden Test Sets**: 16 samples (for DrivAerML)
- **Data Types**: Surface pressure, surface friction (i.e., wallshear stress), volume velocity, volume vorticity, and volume pressure

## Quick Start

### 1. Download Raw Data

Instructions to download the AhmedML dataset: https://caemldatasets.org/ahmedml/
Instructions to download the DrivAerML dataset: https://caemldatasets.org/drivaerml/

### 2. Preprocess the Data
To preprocess either of the two datasets, please use the following script [preprocessing.py](./preprocessing.py)

```bash
python preprocessing.py --dataset={drivaerml,ahmedml} --root=<path to raw downloaded data> --output_dir=<path to output, should be put in the config of the experiments> --subsample_factor=<subsample factor>
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

## Computing the Statistics of the Dataset

AhmedML:
```bash
noether-dataset-stats \
    --dataset_kind=noether.data.datasets.cfd.caeml.AhmedMLDataset \
    --split=train \
    --root=<path to preprocessed ahmedml root> \
    --exclude_attributes=surface_normals,volume_normals,volume_sdf \
    --log_scale=volume_vorticity
```

DrivAerML:
```bash 
noether-dataset-stats \
    --dataset_kind=noether.data.datasets.cfd.caeml.DrivAerMLDataset \
    --split=train \
    --root=<path to preprocessed drivaerml root> \
    --exclude_attributes=surface_normals,volume_normals,volume_sdf \
    --log_scale=volume_vorticity
```
