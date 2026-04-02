# Aero CFD Examples

Training scripts for aerodynamic CFD datasets using the preset-based interface.

## Available examples

| Script | Dataset | Source |
|---|---|---|
| `train_ahmedml.py` | AhmedML (CAEML benchmark) | [CAEML](https://caeml.org/) |
| `train_drivaerml.py` | DrivAerML (CAEML benchmark) | [CAEML](https://caeml.org/) |
| `train_drivaernet.py` | DrivAerNet++ | [DrivAerNet](https://github.com/Mohamedelrefaie/DrivAerNet) |
| `train_emmi_wing.py` | Emmi Wing | Internal |
| `train_shapenet_car.py` | ShapeNet Car | [ShapeNet](https://shapenet.org/) |

Each script contains functions for training different model architectures (AB-UPT, UPT, Transformer, Transolver).

## How to run

From the repository root:

```console
PYTHONPATH=. uv run python examples/aero_cfd/train_emmi_wing.py
```

Before running, update the `DATASET_ROOT` and `OUTPUT_PATH` variables in the script to point to your local data and 
output directories.

To train a different model architecture, edit the `if __name__ == "__main__"` block at the bottom of the script to call 
the desired function (e.g., `train_upt()` instead of `train_abupt()`).
