#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import os
import sys
from pathlib import Path

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from noether.inference.runners.inference_runner import InferenceRunner
from noether.training.cli import setup_hydra

setup_hydra()


@hydra.main(
    config_path=None,
    config_name=None,
    version_base="1.3",
)
def main(inference_config: DictConfig):
    """Main entry point for inference.

    This script is wrapped in a hydra function to allow for easy configuration.
    It supports passing the configuration file as a positional argument or via the --hp flag.

    Example:
        python main_inference.py configs/my_inference_config.yaml
        python main_inference.py --hp configs/my_inference_config.yaml
        python main_inference.py configs/my_inference_config.yaml input_dir=/path/to/run
    """
    # disable hydra changing working directory
    os.chdir(hydra.utils.get_original_cwd())

    # add working directory to PYTHONPATH
    sys.path.insert(0, hydra.utils.get_original_cwd())

    if "input_dir" not in inference_config:
        raise ValueError("input_dir must be specified in the inference configuration")
    input_dir = Path(inference_config.input_dir).resolve()
    hp_resolved_path = input_dir / "hp_resolved.yaml"
    if not hp_resolved_path.exists():
        raise FileNotFoundError(f"hp_resolved.yaml not found in {input_dir}")

    # Load training config as base
    with open(hp_resolved_path) as f:
        train_config = yaml.safe_load(f)

    # Merge: Training config is base, Inference config overrides
    # OmegaConf.merge handles DictConfig and dict
    merged_config = OmegaConf.merge(train_config, inference_config)

    # Force resume from the specified input_dir
    # We prioritize the training run's ID if available, otherwise use folder name
    merged_config.resume_run_id = train_config.get("run_id") or input_dir.name

    # If no stage name is provided, use the one from training
    if "resume_stage_name" not in merged_config or merged_config.resume_stage_name is None:
        merged_config.resume_stage_name = train_config.get("stage_name")

    # If no checkpoint is specified in inference, default to latest
    if "resume_checkpoint" not in inference_config:
        merged_config.resume_checkpoint = "latest"

    # resolve and convert to dict
    config_dict = OmegaConf.to_container(merged_config, resolve=True)

    # run
    InferenceRunner().run(config_dict)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
