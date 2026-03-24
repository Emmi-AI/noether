#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import os
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

_HELP_TEXT = """\
noether-submit-job — validate a training config and submit it as a SLURM job

USAGE
  noether-submit-job --hp <config.yaml> [hydra overrides] [--dry-run]
  noether-submit-job <config.yaml> [hydra overrides] [--dry-run]

ARGUMENTS
  <config.yaml>         Path to the Hydra training configuration file.
                        Can be passed as the first positional argument or
                        via the --hp flag.

  overrides             Hydra-style key=value overrides applied on top of
                        the config file before validation and submission.
                        Examples:
                          seed=42
                          trainer.max_epochs=100
                          +slurm.job_name=my_experiment
                          tracker=disabled

  --dry-run             Print the sbatch command that would be executed
                        without actually submitting the job.

  --help, -h            Show this help message and exit.

DESCRIPTION
  1. Loads and resolves the Hydra config (config file + overrides).
  2. Validates it against the schema declared by config_schema_kind.
  3. Builds an sbatch command from the [slurm] section of the config.
  4. Submits:  sbatch <slurm-flags> --wrap="uv run noether-train --hp <config> [overrides]"

  A [slurm] section is required in the config. Set env_path to source a
  virtual environment inside the job (e.g. env_path: .venv/bin/activate).

SLURM CONFIG (config.yaml → slurm:)
  job_name        Job name shown in squeue (--job-name)
  partition       Target partition, e.g. gpu  (--partition)
  nodes           Number of nodes  (--nodes)
  ntasks_per_node Tasks per node  (--ntasks-per-node)
  cpus_per_task   CPUs per task  (--cpus-per-task)
  gpus_per_node   GPUs per node, e.g. 4 or a100:4  (--gpus-per-node)
  mem             Memory per node, e.g. 64G  (--mem)
  time            Wall-clock limit, e.g. 12:00:00  (--time)
  output          Stdout file path, supports %j %x %u  (--output)
  error           Stderr file path  (--error)
  chdir           Working directory inside the job  (--chdir)
  env_path        Script to source before running, e.g. .venv/bin/activate

EXAMPLES
  # Submit with default config
  noether-submit-job --hp configs/train_shapenet.yaml

  # Override seed and disable tracker
  noether-submit-job --hp configs/train_shapenet.yaml seed=1 tracker=disabled

  # Preview the sbatch command without submitting
  noether-submit-job --hp configs/train_shapenet.yaml --dry-run

  # Override a SLURM field on the fly
  noether-submit-job --hp configs/train_shapenet.yaml +slurm.time=02:00:00
"""

if "--help" in sys.argv or "-h" in sys.argv:
    print(_HELP_TEXT)
    sys.exit(0)

# Strip --dry-run from sys.argv BEFORE setup_hydra() and Hydra see it —
# Hydra errors on any unrecognised flags.
_DRY_RUN: bool = "--dry-run" in sys.argv
if _DRY_RUN:
    sys.argv.remove("--dry-run")

# Capture the config path from argv BEFORE setup_hydra() rewrites sys.argv.
# setup_hydra() replaces --hp <path> with -cp <dir> -cn <name>, so parsing
# must happen here at import time, not inside main().
_RAW_CONFIG_PATH: str | None = None

for idx, arg in enumerate(sys.argv[:-1]):
    if arg == "--hp":
        _RAW_CONFIG_PATH = sys.argv[idx + 1]
        break

if _RAW_CONFIG_PATH is None and len(sys.argv) > 1 and sys.argv[1].endswith(".yaml"):
    _RAW_CONFIG_PATH = sys.argv[1]


def validate_config(config: DictConfig) -> ConfigSchema:
    """Validate the configuration using the specified schema.

    Args:
        config: The Hydra configuration to validate

    Returns:
        The validated configuration schema

    Raises:
        ValueError: If the configuration is missing required fields
        RuntimeError: If the schema class cannot be loaded
    """
    config_dict = yaml.safe_load(OmegaConf.to_yaml(config, resolve=True))

    config_schema_kind = config_dict.get("config_schema_kind")
    if not config_schema_kind:
        raise ValueError("Configuration must specify 'config_schema_kind'")

    print(f"Validating configuration with schema: {config_schema_kind}")
    config_schema_class = class_constructor_from_class_path(config_schema_kind)
    validated_config: ConfigSchema = config_schema_class(**config_dict)
    print("Configuration validated successfully")
    return validated_config


def _find_config_path() -> str:
    """Return the config file path, resolved to an absolute path.

    Reads from the pre-captured _RAW_CONFIG_PATH (set before setup_hydra()
    mutates sys.argv), or falls back to reconstructing from -cp/-cn.
    """
    if _RAW_CONFIG_PATH is not None:
        return str(Path(_RAW_CONFIG_PATH).resolve())

    # Fallback: reconstruct from -cp / -cn (post-setup_hydra argv)
    config_dir = None
    config_name = None

    i = 0
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "-cp" and i + 1 < len(sys.argv):
            config_dir = sys.argv[i + 1]
            i += 2
            continue
        if arg == "-cn" and i + 1 < len(sys.argv):
            config_name = sys.argv[i + 1]
            i += 2
            continue
        i += 1

    if config_dir and config_name:
        config_path = str(Path(config_dir) / config_name)
        if not config_path.endswith((".yaml", ".yml")):
            config_path += ".yaml"
        return str(Path(config_path).resolve())

    print(f"Error: Could not determine config path from arguments\nsys.argv: {sys.argv}")
    sys.exit(1)


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

    Validates a Hydra config and submits a training job via ``sbatch``.

    Example:
    .. code-block:: bash

       noether-submit-job --hp configs/train_shapenet.yaml +seed=1 tracker=disabled
       noether-submit-job --hp configs/train_shapenet.yaml --dry-run
    """
    print("Starting job submission process")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())

    try:
        validated_config = validate_config(config)
    except Exception as e:
        print(f"Configuration validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    if validated_config.slurm is None:
        print(
            "Error: SLURM configuration is required. Please specify the 'slurm' section in your config.",
            file=sys.stderr,
        )
        sys.exit(1)

    slurm_config: SlurmConfig = validated_config.slurm

    # Build the sbatch arguments from the SLURM config (chdir is included here)
    sbatch_args = slurm_config.to_srun_args()
    print(f"SLURM args: {sbatch_args}")

    config_path = _find_config_path()
    print(f"Config path: {config_path}")

    train_cmd = f"uv run noether-train --hp {config_path}"
    hydra_overrides = _collect_hydra_overrides()

    if hydra_overrides:
        train_cmd += " " + " ".join(hydra_overrides)
        print(f"Hydra overrides: {hydra_overrides}")

    source_cmd = ""
    if slurm_config.env_path:
        print(f"Sourcing environment from: {slurm_config.env_path}")
        source_cmd = f"source {slurm_config.env_path};"

    full_cmd = source_cmd + f'sbatch {sbatch_args} --wrap="{train_cmd}"'

    if _DRY_RUN:
        print(f"[dry-run] Would execute:\n  {full_cmd}")
        sys.exit(0)

    print(f"Executing: {full_cmd}")
    result = subprocess.run(full_cmd, shell=True)
    sys.exit(result.returncode)


if __name__ == "__main__":
    setup_hydra()
    main()
