# AB-UPT Showcase

Train, evaluate, and visualize the Anchored-Branched Universal Physics Transformer (AB-UPT) on the DrivAerML automotive aerodynamics dataset.

## Quick Start

All commands must be run from the `recipes/aero_cfd/` directory.

### Train

```bash
cd recipes/aero_cfd/

# Small model — fast iteration / smoke tests
python -m showcase.cli train \
  --dataset-root /path/to/drivaerml \
  --output-path /path/to/outputs

# Scaled model on GPU (hidden_dim=384, 6 heads, 6 decoder blocks/domain)
python -m showcase.cli train \
  --dataset-root /path/to/drivaerml \
  --output-path /path/to/outputs \
  --model-size scaled \
  --accelerator gpu

# Scaled model on Apple Silicon (reduced point budget)
python -m showcase.cli train \
  --dataset-root /path/to/drivaerml \
  --output-path /path/to/outputs \
  --model-size scaled_mps \
  --accelerator mps

# With experiment tracking
python -m showcase.cli train \
  --dataset-root /path/to/drivaerml \
  --output-path /path/to/outputs \
  --tracker wandb \
  --tracker-project my-aero-project
```

### Evaluate

```bash
# Anchor-resolution metrics + save predictions
python -m showcase.cli evaluate \
  --dataset-root /path/to/drivaerml \
  --output-path /path/to/outputs \
  --run-id 2026-04-09_abc12

# Dense query inference (higher-resolution predictions)
python -m showcase.cli evaluate \
  --dataset-root /path/to/drivaerml \
  --output-path /path/to/outputs \
  --run-id 2026-04-09_abc12 \
  --query-inference \
  --num-inference-surface-points 20000

# With VTK export and force coefficients
python -m showcase.cli evaluate \
  --dataset-root /path/to/drivaerml \
  --output-path /path/to/outputs \
  --run-id 2026-04-09_abc12 \
  --query-inference \
  --export-vtk \
  --compute-forces

# Select a specific checkpoint
python -m showcase.cli evaluate \
  --dataset-root /path/to/drivaerml \
  --output-path /path/to/outputs \
  --run-id 2026-04-09_abc12 \
  --checkpoint best_model.loss.test.total
```

### Export to VTK (standalone)

```bash
python -m showcase.cli export-vtk \
  --predictions-path /path/to/outputs/2026-04-09_abc12/eval/predictions/sample_0000.pt \
  --output-path result.vtp \
  --domain surface
```

Open the `.vtp` files in [ParaView](https://www.paraview.org/) and color by field
(e.g., `surface_pressure`, `surface_friction`).

## Model Sizes

| Size | Hidden Dim | Heads | Decoder Blocks | Geometry Points | Supernodes | Surface Anchors | Volume Anchors |
|------|-----------|-------|----------------|-----------------|------------|-----------------|----------------|
| `small` | 192 | 3 | 2 | 16,384 | 1,024 | 512 | 512 |
| `scaled` | 384 | 6 | 6 | 125,000 | 32,000 | 16,000 | 32,000 |
| `scaled_mps` | 384 | 6 | 6 | 16,384 | 4,096 | 2,048 | 4,096 |

All sizes use the same physics block pattern: `perceiver -> self -> cross -> self -> cross -> self`.

- **`small`** — Fast iteration and smoke tests. Default for CPU.
- **`scaled`** — Full research configuration for GPU (matches `train_drivaerml_ab-upt_scale.yaml`).
- **`scaled_mps`** — Same architecture as `scaled` with reduced point budget to fit Apple Silicon MPS memory and INT32 index limits.

## Query-Based Inference

By default, evaluation predicts at the same anchor-point resolution used during
training.  With `--query-inference`, the model additionally predicts at extra
query positions, processed in chunks to manage memory:

```
--query-inference --num-inference-surface-points 20000 --num-inference-volume-points 20000
```

The chunk size defaults to `num_surface_anchor_points` from the model size config,
matching the workload the model handles during a single training step.

## Force Coefficients

`--compute-forces` computes ground-truth and predicted drag/lift coefficients (Cd/Cl)
for each sample in the evaluation split.  Requires per-run reference data:

- `surface_normal_vtp.pt` — cell normals (included in the dataset)
- `surface_area_vtp.pt` — cell areas scaled by the subsample factor (precomputed from the original VTP mesh)
- `geo_ref_<N>.csv` — per-run reference area (optional, falls back to DrivAerML defaults)

Results are saved to `forces.csv` in the predictions directory.

## Results

_Results to be populated after benchmark runs._

| Model | Surface Pressure L2 | Volume Velocity L2 | Cd Error | Cl Error |
|-------|--------------------|--------------------|----------|----------|
| `scaled` | -- | -- | -- | -- |
| `scaled_mps` | -- | -- | -- | -- |
