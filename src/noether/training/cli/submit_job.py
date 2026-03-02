#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

"""Submit SLURM jobs for training with config validation."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from noether.core.factory import class_constructor_from_class_path
from noether.core.schemas.schema import ConfigSchema
from noether.training.cli import setup_hydra

setup_hydra()


def validate_config(config: DictConfig) -> ConfigSchema:
    """Validate the configuration using the specified schema.

    Args:
        config: The Hydra configuration to validate

    Returns:
        The validated configuration schema

    Raises:
        ValidationError: If the configuration is invalid
    """
    config_dict = yaml.safe_load(OmegaConf.to_yaml(config, resolve=True))

    config_schema_kind = config_dict.get("config_schema_kind")
    if not config_schema_kind:
        raise ValueError("Configuration must specify 'config_schema_kind'")

    print(f"Validating configuration with schema: {config_schema_kind}")
    config_schema_class = class_constructor_from_class_path(config_schema_kind)
    # This will raise a ValidationError if the config is invalid
    validated_config: ConfigSchema = config_schema_class(**config_dict)
    print("✓ Configuration validated successfully")

    return validated_config


@hydra.main(
    config_path=None,
    config_name=None,
    version_base="1.3",
)
def main(config: DictConfig):
    """Main entry point for SLURM job submission.

    This script validates a Hydra config and submits a SLURM job.

    Example:
    .. code-block:: bash

       noether-submit-job --hp configs/train_shapenet.yaml +seed=1 tracker=disabled \\
           --job-name shapenet_exp --gpus-per-node 2 --mem 128GB

       # With experiment file for array jobs
       noether-submit-job --hp configs/train_shapenet.yaml \\
           --job-name shapenet_sweep --array 1-20%5 \\
           --experiment-file jobs/experiments/shapenet_experiments.txt
    """
    # Parse additional arguments for SLURM configuration
    parser = argparse.ArgumentParser(
        description="Submit SLURM job with config validation",
        add_help=False,  # Avoid conflict with Hydra
    )

    # Parse known args (SLURM params) and let Hydra handle the rest
    args, unknown_args = parser.parse_known_args()

    # Extract config path from sys.argv
    # Hydra converts --hp to -cp (config path) and -cn (config name)
    config_path = None
    config_dir = None
    config_name = None

    i = 0
    while i < len(sys.argv):
        arg = sys.argv[i]

        # Check for --hp (before Hydra processing)
        if arg == "--hp" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            i += 2
            continue

        # Check for -cp (config path) - after Hydra processing
        if arg == "-cp" and i + 1 < len(sys.argv):
            config_dir = sys.argv[i + 1]
            i += 2
            continue

        # Check for -cn (config name) - after Hydra processing
        if arg == "-cn" and i + 1 < len(sys.argv):
            config_name = sys.argv[i + 1]
            i += 2
            config_overrides = sys.argv[i + 2 :]
            continue

        i += 1

    # Reconstruct config_path from Hydra's -cp and -cn if needed
    if not config_path and config_dir and config_name:
        config_path = str(Path(config_dir) / config_name)
        # Add .yaml extension if not present
        if not config_path.endswith(".yaml") and not config_path.endswith(".yml"):
            config_path += ".yaml"

    if not config_path:
        print("Error: Could not determine config path from arguments")
        print(f"sys.argv: {sys.argv}")
        sys.exit(1)

    config_overrides = []

    try:
        validated_config = validate_config(config)
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        sys.exit(1)

    if validated_config.slurm is None:
        raise ValueError(
            "SLURM configuration is required for job submission. Please specify the 'slurm' section in your config."
        )

    if validated_config.slurm.array and not validated_config.slurm.experiment_file:
        raise ValueError(
            "SLURM array specified without experiment file. Config overrides will be ignored in array jobs."
        )

    if validated_config.slurm.experiment_file and not validated_config.slurm.array:
        raise ValueError(
            "Experiment file specified without SLURM array. The experiment file is only used for array jobs."
        )

    # Read the job template and fill in variables from the slurm config
    slurm = validated_config.slurm
    template_path = Path(__file__).parent / "noether_train.job"
    template = template_path.read_text()

    script = template.format(
        cpus_per_task=slurm.cpus_per_task,
        partition=slurm.partition,
        gpus_per_node=slurm.gpus_per_node,
        tasks_per_node=slurm.ntasks_per_node,
        mem=slurm.mem,
        output=slurm.output,
        nice=slurm.nice,
        array_string=slurm.array,
        env_name=slurm.export or "",
        hp_file=config_path,
        experiment_file=slurm.experiment_file,
        chdir=slurm.chdir,
        source=slurm.source,
        nodes=slurm.nodes or 1,
    )

    # Write the filled-in script to a temp file and submit via sbatch
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=validated_config.slurm.job_name + "_" if validated_config.slurm.job_name else "",
        suffix=".job",
        delete=False,
    ) as f:
        f.write(script)
        job_script_path = f.name

    print(f"Generated SLURM script ({job_script_path}):\n{script}")

    result = subprocess.run(["sbatch", job_script_path], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"sbatch failed: {result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
