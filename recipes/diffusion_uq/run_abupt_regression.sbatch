#!/bin/bash
#SBATCH --job-name=abupt_reg
#SBATCH --cpus-per-task=28
#SBATCH --mem=250GB
#SBATCH --gpus-per-node=1
#SBATCH --time=1-0
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd /home/ggalletti/exp/noether

# train regression AB-UPT via the canonical aero_cfd recipe
# (examples/aero_cfd/train_drivaerml.py::train_abupt) with cuda + our paths +
# a longer max_epochs than the example's default of 2.
python - <<'PY'
from examples.aero_cfd import DrivAerMLPreset
from noether.training.runners import HydraRunner

# patch AeroABUPT.forward: upstream returns (preds, kv_cache) tuple post-PR#131
# but example still expects plain dict — unwrap so WeightedLossTrainer sees dict keys.
from noether.modeling.models.aerodynamics import AeroABUPT
def _abupt_forward(self, **kwargs):
    preds, _kv = self.backbone(**kwargs)
    return preds
AeroABUPT.forward = _abupt_forward

DATASET_ROOT = "/nfs-gpu/research/datasets/drivaerml/preprocessed/subsampled_10x"
OUTPUT_PATH = "./outputs/abupt_regression"
FIELD_WEIGHTS = {
    "surface_pressure": 1.0, "surface_friction": 1.0,
    "volume_pressure": 1.0, "volume_velocity": 1.0, "volume_vorticity": 1.0,
}

import os
from noether.core.providers.path import PathProvider

preset = DrivAerMLPreset()
# match diffusion geometry budget (16384 supernodes / 65536 sampled points) so
# the regression baseline gets the same geometry encoder capacity → fair comparison.
model_kind = "noether.modeling.models.aerodynamics.AeroABUPT"
preset.pipeline_model_overrides[model_kind]["num_geometry_supernodes"] = 16384
preset.pipeline_model_overrides[model_kind]["num_geometry_points"] = 65536

config = preset.build_config(
    model_kind=model_kind,
    model_params=dict(
        hidden_dim=192,
        geometry_depth=6,
        physics_blocks=["perceiver"] + ["shared", "cross"] * 5,
    ),
    trainer_kind="noether.training.trainers.WeightedLossTrainer",
    trainer_params=dict(field_weights=FIELD_WEIGHTS),
    dataset_root=DATASET_ROOT,
    output_path=OUTPUT_PATH,
    datasets=["train", "val", "test"],
    max_epochs=100,
    accelerator="gpu",
)

# run id = <SLURM_JOB_ID>_<date>_<hash> so wandb/output dirs trace back to the job
run_id = PathProvider.generate_run_id()
job_id = os.environ.get("SLURM_JOB_ID")
if job_id:
    run_id = f"{job_id}_{run_id}"
config.run_id = run_id
config.name = f"{config.model.name}_{run_id}"

# wandb tracker
from noether.core.schemas.trackers import WandBTrackerSchema
config.tracker = WandBTrackerSchema(
    kind="noether.core.trackers.WandBTracker",
    project="steady_diffusion_gg",
    entity="emmi-ai",
    mode="online",
)

HydraRunner().main(device="cuda", config=config)
PY
