#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Submit SLURM jobs for training with config validation."""

import logging
import subprocess
import sys
from pathlib import Path

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from noether.core.factory import class_constructor_from_class_path
from noether.core.schemas.schema import ConfigSchema
from noether.core.schemas.slurm import SlurmConfig
from noether.training.cli import setup_hydra

logger = logging.getLogger(__name__)

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

    logger.info(f"Validating configuration with schema: {config_schema_kind}")
    config_schema_class = class_constructor_from_class_path(config_schema_kind)
    validated_config: ConfigSchema = config_schema_class(**config_dict)
    logger.info("Configuration validated successfully")

    return validated_config


def _find_config_path() -> str:
    """Extract the config file path from sys.argv."""
    config_path = None
    config_dir = None
    config_name = None

    i = 0
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--hp" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            i += 2
            continue
        if arg == "-cp" and i + 1 < len(sys.argv):
            config_dir = sys.argv[i + 1]
            i += 2
            continue
        if arg == "-cn" and i + 1 < len(sys.argv):
            config_name = sys.argv[i + 1]
            i += 2
            continue
        i += 1

    if not config_path and config_dir and config_name:
        config_path = str(Path(config_dir) / config_name)
        if not config_path.endswith((".yaml", ".yml")):
            config_path += ".yaml"

    if not config_path:
        logger.error(f"Error: Could not determine config path from arguments\nsys.argv: {sys.argv}")
        sys.exit(1)

    return config_path


def _collect_hydra_overrides() -> list[str]:
    """Collect Hydra overrides from sys.argv (arguments that contain '=' and aren't flags)."""
    overrides: list[str] = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in ("--hp", "-cp", "-cn"):
            skip_next = True
            continue
        if "=" in arg and not arg.startswith("-") and not arg.startswith("hydra"):
            overrides.append(arg)

    return overrides


@hydra.main(
    config_path=None,
    config_name=None,
    version_base="1.3",
)
def main(config: DictConfig):
    """Main entry point for SLURM job submission.

    Validates a Hydra config and submits a training job via ``srun``.

    Example:
    .. code-block:: bash

       noether-submit-job --hp configs/train_shapenet.yaml +seed=1 tracker=disabled
    """

    try:
        validated_config = validate_config(config)
    except Exception as e:
        logger.info(f"Configuration validation failed: {e}")
        sys.exit(1)

    if validated_config.slurm is None:
        raise ValueError(
            "SLURM configuration is required for job submission. Please specify the 'slurm' section in your config."
        )

    slurm_config: SlurmConfig = validated_config.slurm

    # Build the sbatch command
    sbatch_args = slurm_config.to_srun_args()

    config_path = _find_config_path()

    train_cmd = f"uv run noether-train --hp {config_path}"
    hydra_overrides = _collect_hydra_overrides()

    if hydra_overrides:
        # Replace single quotes with escaped single quotes (\') for shell compatibility
        escaped_overrides = [o.replace("'", "'") for o in hydra_overrides]
        train_cmd += " " + " ".join(escaped_overrides)

    # Build the wrapped command (runs inside the job)
    wrap_cmd = train_cmd

    cd_cmd = ""
    source_cmd = ""

    if slurm_config.chdir:
        logger.info(f"Changing directory to: {slurm_config.chdir}")
        cd_cmd = f"cd {slurm_config.chdir};"

    if slurm_config.env_path:
        logger.info(f"Sourcing environment from: {slurm_config.env_path}")
        source_cmd = f"source {slurm_config.env_path};"

    full_cmd = cd_cmd + source_cmd + f'sbatch {sbatch_args} --wrap="{wrap_cmd}"'

    logger.info(f"Executing: {full_cmd}")
    result = subprocess.run(full_cmd, shell=True)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
