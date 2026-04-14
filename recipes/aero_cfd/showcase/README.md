# AB-UPT Showcase

Train, evaluate, and visualize the Anchored-Branched Universal Physics Transformer (AB-UPT) on the DrivAerML automotive aerodynamics dataset.

## Dataset

The DrivAerML dataset (subsampled 10x) is hosted on HuggingFace:
[EmmiAI/DrivAerML_subsampled_10x](https://huggingface.co/datasets/EmmiAI/DrivAerML_subsampled_10x)

### Download with `noether-data` CLI

```bash
# Full dataset snapshot
noether-data huggingface snapshot EmmiAI/DrivAerML_subsampled_10x /path/to/drivaerml
```

See `noether.io.cli` for additional options (verification, manifests, parallel downloads).

### Download with huggingface_hub

```bash
uv pip install huggingface_hub

huggingface-cli download EmmiAI/DrivAerML_subsampled_10x \
  --repo-type dataset \
  --local-dir /path/to/drivaerml
```

### Download with Python

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="EmmiAI/DrivAerML_subsampled_10x",
    repo_type="dataset",
    local_dir="/path/to/drivaerml",
)
```

## Quick Start

All commands must be run from the `recipes/aero_cfd/` directory with `recipes/` on the Python path:

```bash
cd recipes/aero_cfd/
export PYTHONPATH=$(git -C ../.. rev-parse --show-toplevel)/recipes:$PYTHONPATH
```

### Train

```bash

# Small model -- fast iteration / smoke tests
python -m showcase.cli train \
  --dataset-root /path/to/drivaerml \
  --output-path /path/to/outputs

# Scaled model on GPU (hidden_dim=384, 6 heads, 6 decoder blocks/domain)
python -m showcase.cli train \
  --dataset-root /path/to/drivaerml \
  --output-path /path/to/outputs \
  --model-size scaled \
  --accelerator gpu \
  --precision float16

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

Open the `.vtp` files in [ParaView](https://www.paraview.org/) and color by field (e.g., `surface_pressure`, 
`surface_friction`).

## Model Sizes

| Size | Hidden Dim | Heads | Decoder Blocks | Geometry Points | Supernodes | Surface Anchors | Volume Anchors |
|------|-----------|-------|----------------|-----------------|------------|-----------------|----------------|
| `small` | 192 | 3 | 2 | 16,384 | 1,024 | 512 | 512 |
| `scaled` | 384 | 6 | 6 | 125,000 | 32,000 | 16,000 | 32,000 |
| `scaled_mps` | 384 | 6 | 6 | 16,384 | 4,096 | 2,048 | 4,096 |

All sizes use the same physics block pattern: `perceiver -> self -> cross -> self -> cross -> self`.

- **`small`** -- Fast iteration and smoke tests. Default for CPU.
- **`scaled`** -- Full research configuration for GPU. Use `--precision float16` for ~2x speedup on supported hardware.
- **`scaled_mps`** -- Same architecture as `scaled` with reduced point budget to fit Apple Silicon MPS memory and INT32 index limits (on a 32GB machine).

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

- `surface_normal_vtp.pt` -- cell normals (included in the dataset)
- `surface_area_vtp.pt` -- cell areas scaled by the subsample factor (precomputed from the original VTP mesh)
- `geo_ref_<N>.csv` -- per-run reference area (optional, falls back to DrivAerML defaults)

Results are saved to `forces.csv` in the predictions directory.

## Results

_Results to be populated after benchmark runs._

| Model | Surface Pressure L2 | Volume Velocity L2 | Cd Error | Cl Error |
|-------|--------------------|--------------------|----------|----------|
| `scaled` | -- | -- | -- | -- |
| `scaled_mps` | -- | -- | -- | -- |
