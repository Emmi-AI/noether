#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import getpass
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import submitit
import yaml
from hydra import compose, initialize_config_dir
from hydra._internal.core_plugins.basic_sweeper import BasicSweeper
from hydra.core.global_hydra import GlobalHydra
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, field_validator, model_validator

from noether.core.factory import class_constructor_from_class_path

if TYPE_CHECKING:
    from noether.core.schemas.schema import ConfigSchema

# SLURM patterns that require a running job and cannot be resolved at submission time.
_UNSUPPORTED_FOLDER_PATTERNS: frozenset[str] = frozenset({"%j", "%J", "%A", "%a", "%N", "%x"})


class SlurmConfig(BaseModel):
    """Configuration for SLURM job submission via :mod:`submitit`.

    Field names mirror the keyword arguments accepted by
    :meth:`submitit.AutoExecutor.update_parameters`. All fields are optional
    and default to ``None``, meaning the cluster default is used.

    Note:
        Job stdout/stderr is owned by submitit and written to ``<folder>/<job_id>_log.out``
        / ``<folder>/<job_id>_log.err``. Use the ``folder`` field to control where these
        files land. SLURM ``--output``/``--error`` directives are intentionally not
        exposed; pass them via ``slurm_additional_parameters`` if you really need to
        override submitit's defaults (this disables ``job.stdout()`` helpers).
    """

    model_config = {"extra": "forbid"}

    # --- Executor constructor argument ---
    folder: str = "submitit_logs"
    """Directory where submitit writes the job script, pickled task, and stdout/stderr logs.
    Per-job files are named ``<job_id>_log.out`` etc. inside this directory.
    This is also used as the default ``output_path`` for training runs (see
    :attr:`ConfigSchema.output_path`).

    Supports ``%u`` (current username) interpolation, e.g.
    ``/home/%u/logs/experiment``. SLURM job-time patterns like ``%j`` are
    **not** supported because submitit needs the directory to exist before
    submission."""

    # --- AutoExecutor-generic parameters (mapped to update_parameters as-is) ---
    name: str | None = None
    """Job name (SLURM ``--job-name``)."""

    nodes: int | None = None
    """Number of nodes to allocate."""

    tasks_per_node: int | None = None
    """Number of tasks per allocated node."""

    cpus_per_task: int | None = None
    """Number of CPUs per task."""

    gpus_per_node: int | str | None = None
    """GPUs per node. Accepts a count or ``type:count`` (e.g. ``"a100:4"``)."""

    mem_gb: float | None = None
    """Memory per node in gigabytes."""

    timeout_min: int = 0
    """Wall-clock limit in minutes. Use 0 for no time limit"""

    stderr_to_stdout: bool | None = None
    """If True, merge stderr into stdout."""

    # --- Slurm-specific parameters (forwarded with the ``slurm_`` prefix) ---
    slurm_partition: str | None = None
    """Partition to submit the job to."""

    slurm_array_parallelism: int | None = None
    """Maximum number of array tasks running concurrently (SLURM ``%N`` in ``--array``)."""

    slurm_setup: list[str] | None = None
    """Shell commands run inside the job before the main command, e.g.
    ``["source .venv/bin/activate"]``."""

    slurm_additional_parameters: dict[str, Any] | None = None
    """Escape hatch for SLURM directives not exposed as first-class fields, e.g.
    ``{"nice": 0, "reservation": "my_res", "chdir": "/work"}``. Keys are passed as
    ``--key=value`` to ``sbatch``."""

    @field_validator("folder")
    @classmethod
    def _resolve_folder_patterns(cls, value: str) -> str:
        """Resolve ``%u`` to the current username; reject job-time SLURM patterns."""
        for pat in _UNSUPPORTED_FOLDER_PATTERNS:
            if pat in value:
                raise ValueError(
                    f"SLURM pattern '{pat}' in folder is not supported — submitit needs "
                    "the directory to exist before submission. Use %u (username) instead, "
                    "or remove the pattern."
                )
        return value.replace("%u", getpass.getuser())

    @field_validator("gpus_per_node")
    @classmethod
    def _validate_gpu_spec(cls, value: str | int | None) -> str | int | None:
        if value is None or isinstance(value, int):
            if isinstance(value, int) and value < 0:
                raise ValueError(f"gpus_per_node must be non-negative, got {value}.")
            return value
        if not re.match(r"^(\w+:)?\d+$", value):
            raise ValueError(
                f"Invalid gpus_per_node spec: '{value}'. Expected a count or 'type:count' (e.g. '2', 'a100:4')."
            )
        return value

    @model_validator(mode="after")
    def _set_tasks_per_node_if_gpus_set(self):
        """If gpus_per_node is set but tasks_per_node isn't, set tasks_per_node = gpus_per_node."""
        if self.gpus_per_node is not None and self.tasks_per_node is None:
            gpus = self.gpus_per_node if isinstance(self.gpus_per_node, int) else int(self.gpus_per_node.split(":")[-1])
            self.tasks_per_node = gpus
        return self

    def to_executor_kwargs(self) -> tuple[str, dict[str, Any]]:
        """Return ``(folder, update_parameters_kwargs)`` for :class:`submitit.AutoExecutor`.

        Generic fields are passed under their bare name; everything else keeps its
        ``slurm_`` prefix so submitit routes it to the slurm executor.

        Returns:
            A tuple ``(folder, kwargs)`` where ``folder`` is the executor's log directory
            and ``kwargs`` is the dict to splat into ``executor.update_parameters(**kwargs)``.
        """
        params: dict[str, Any] = {}
        for name, value in self:
            if name == "folder" or value is None:
                continue
            params[name] = value
        return self.folder, params


_HELP_TEXT = """\
noether-train-submit-job — validate a training config and submit it to SLURM via submitit

USAGE
  noether-train-submit-job --hp <config.yaml> [hydra overrides] [-m] [--dry-run]
  noether-train-submit-job <config.yaml> [hydra overrides] [-m] [--dry-run]

ARGUMENTS
  <config.yaml>     Path to the Hydra training configuration file. Can be passed as
                    the first positional argument or via the --hp flag.

  overrides         Hydra-style key=value overrides applied on top of the config
                    before validation and submission. Examples:
                      seed=42
                      trainer.max_epochs=100
                      tracker=disabled
                      seed=1,2,3              (sweep — requires --multirun)

  -m, --multirun    Enumerate all sweep combinations (e.g. seed=1,2,3) and submit
                    them as a single SLURM array job. The cross-product is
                    validated upfront on the login node before anything is sent.

  --dry-run         Print the submitit parameters and per-task command(s) without
                    submitting anything.

  --help, -h        Show this help message and exit.

DESCRIPTION
  1. Parses sweep overrides and, for --multirun, enumerates the cross-product.
  2. For every combination, composes the Hydra config and validates it against
     the schema declared by ``config_schema_kind`` (the ``slurm`` section is
     required and must NOT vary across the sweep).
  3. Builds a ``submitit.AutoExecutor`` from the validated [slurm] section and
     submits the job(s). Multirun submissions are wrapped in
     ``executor.batch()`` so SLURM sees a single array job.

LOG FILES
  Submitit owns the job's stdout/stderr. They are written to:
      <slurm.folder>/<job_id>_log.out
      <slurm.folder>/<job_id>_log.err
  For array jobs, ``<job_id>`` becomes ``<array_master_id>_<task_idx>``.
  SLURM ``--output`` / ``--error`` are intentionally not exposed; if you must
  override them, pass them via ``slurm.slurm_additional_parameters``.

SLURM CONFIG (config.yaml → slurm:)
  folder                       Submitit log/script directory (default ``submitit_logs``).
                               Also used as the default ``output_path`` for training
                               runs when ``output_path`` is omitted from the config.
  name                         Job name (--job-name)
  slurm_partition              Partition, e.g. ``gpu``
  nodes                        Number of nodes
  tasks_per_node               Tasks per node
  cpus_per_task                CPUs per task
  gpus_per_node                GPUs per node, e.g. 4 or ``a100:4``
  mem_gb                       Memory per node in gigabytes (float)
  timeout_min                  Wall-clock limit in minutes (int)
  slurm_array_parallelism      Max concurrent array tasks
  slurm_setup                  List of shell commands run before the main command,
                               e.g. ``["source .venv/bin/activate"]``
  slurm_additional_parameters  Dict for any other sbatch directive
                               (``nice``, ``reservation``, ``chdir``, ``account``, ...)

EXAMPLES
  # Single submission
  noether-train-submit-job --hp configs/train_shapenet.yaml

  # Override seed and disable tracker
  noether-train-submit-job --hp configs/train_shapenet.yaml seed=1 tracker=disabled

  # Sweep over 3 seeds and 2 learning rates → one 6-task SLURM array
  noether-train-submit-job --hp configs/train_shapenet.yaml -m seed=1,2,3 trainer.lr=1e-3,1e-4

  # Preview without submitting
  noether-train-submit-job --hp configs/train_shapenet.yaml --dry-run
"""


def _parse_argv(argv: list[str]) -> tuple[str, list[str], bool, bool]:
    """Split argv into (config_path, hydra_overrides, multirun, dry_run).

    Args:
        argv: The argv list to parse, typically ``sys.argv``.

    Returns:
        A tuple ``(config_path, overrides, multirun, dry_run)``.

    Raises:
        SystemExit: If no config path was provided.
    """
    config_path: str | None = None
    overrides: list[str] = []
    multirun = False
    dry_run = False

    args = list(argv[1:])
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--multirun", "-m"):
            multirun = True
        elif arg == "--dry-run":
            dry_run = True
        elif arg == "--hp":
            if i + 1 >= len(args):
                print("Error: --hp requires a path argument", file=sys.stderr)
                sys.exit(1)
            config_path = args[i + 1]
            i += 1
        elif config_path is None and arg.endswith((".yaml", ".yml")) and not arg.startswith("-"):
            config_path = arg
        elif "=" in arg and not arg.startswith("-"):
            overrides.append(arg)
        else:
            print(f"Error: unrecognised argument: {arg}", file=sys.stderr)
            sys.exit(1)
        i += 1

    if config_path is None:
        print("Error: no config file provided. Use --hp <path> or pass it as the first argument.", file=sys.stderr)
        sys.exit(1)

    return config_path, overrides, multirun, dry_run


def _expand_sweeps(overrides: list[str], multirun: bool) -> list[list[str]]:
    """Expand sweep overrides into the list of per-run override sets.

    In multirun mode, ``a=1,2 b=x,y`` becomes 4 combinations. Otherwise the
    overrides are returned as a single combination (sweep syntax raises).

    Args:
        overrides: Hydra-style override strings.
        multirun: Whether to expand comma-separated sweeps.

    Returns:
        A list of override-lists, one per submitted task.
    """
    parser = OverridesParser.create()
    parsed = parser.parse_overrides(overrides)

    if not multirun:
        for ov in parsed:
            if ov.is_sweep_override():
                raise ValueError(
                    f"Sweep override '{ov.input_line}' requires --multirun (-m). "
                    "Without --multirun, only single-value overrides are allowed."
                )
        return [overrides]

    batches = BasicSweeper.split_arguments(parsed, max_batch_size=None)
    # split_arguments returns a list of batches; with max_batch_size=None there is exactly one.
    return batches[0]


def validate_config(config: DictConfig) -> ConfigSchema:
    """Validate the configuration using the schema declared by ``config_schema_kind``.

    Args:
        config: The composed Hydra configuration to validate.

    Returns:
        The validated configuration schema instance.

    Raises:
        ImportError: If the schema class cannot be imported.
        ValidationError: If the configuration does not satisfy the schema.
    """
    # Lazy import: ``ConfigSchema`` references ``SlurmConfig`` from this module,
    # so importing it at module scope creates a circular import.
    from noether.core.schemas.schema import ConfigSchema

    config_dict = yaml.safe_load(OmegaConf.to_yaml(config, resolve=True))

    config_schema_class: Any = ConfigSchema
    config_schema_kind = config_dict.get("config_schema_kind")
    if config_schema_kind is not None:
        print(f"Validating configuration with schema: {config_schema_kind}")
        config_schema_class = class_constructor_from_class_path(config_schema_kind)
    validated_config: ConfigSchema = config_schema_class(**config_dict)
    return validated_config


def _compose_config(config_path: Path, overrides: list[str]) -> DictConfig:
    """Compose a Hydra config from a yaml file and a list of overrides.

    Each call uses a fresh Hydra context, so this is safe to call repeatedly
    in a loop over sweep combinations.
    """
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(config_path.parent)):
        return compose(config_name=config_path.stem, overrides=overrides)


def _validate_all_combos(config_path: Path, combos: list[list[str]]) -> SlurmConfig:
    """Validate every sweep combination upfront and return the (shared) slurm config.

    Args:
        config_path: Absolute path to the base config yaml.
        combos: List of override-lists, one per sweep combination.

    Returns:
        The :class:`SlurmConfig` from the first combo. The slurm section must be
        identical across combos; sweeping over slurm fields is rejected.

    Raises:
        SystemExit: On validation failure or if the slurm section varies.
    """
    first_slurm: SlurmConfig | None = None
    for idx, combo in enumerate(combos, start=1):
        label = f"[{idx}/{len(combos)}]"
        print(f"{label} validating overrides: {combo if combo else '(none)'}")
        try:
            cfg = _compose_config(config_path, combo)
            validated = validate_config(cfg)
        except Exception as e:
            print(f"{label} configuration validation failed: {e}", file=sys.stderr)
            sys.exit(1)

        if validated.slurm is None:
            print(
                f"{label} Error: SLURM configuration is required. Please specify a 'slurm' section in your config.",
                file=sys.stderr,
            )
            sys.exit(1)

        if first_slurm is None:
            first_slurm = validated.slurm
        elif validated.slurm.model_dump() != first_slurm.model_dump():
            print(
                f"{label} Error: sweeping over fields under 'slurm' is not supported "
                "(the SLURM allocation must be identical across all array tasks).",
                file=sys.stderr,
            )
            sys.exit(1)

    assert first_slurm is not None  # combos is guaranteed non-empty
    print(f"All {len(combos)} configuration(s) validated successfully")
    return first_slurm


def _train_entrypoint(config_path: str, overrides: list[str], cwd: str) -> None:
    """In-process training entrypoint executed by the submitit worker.

    Replaces the previous ``submitit.helpers.CommandFunction`` approach (shelling
    out via ``subprocess.Popen``). With the subprocess approach, a SLURM cgroup
    OOM kill targets the largest memory consumer (the trainer or one of its
    multiprocessing children); the submitit wrapper survives, sits in
    ``copy_process_streams`` reading broken pipes, and ignores any SIGTERM SLURM
    sends to clean up (see ``submitit.core.job_environment.SignalHandler.bypass``)
    — so the step lingers until walltime. Running training in-process means the
    wrapper dies with the trainer, ``srun`` exits, and the step ends.

    ``main_train.main`` is invoked (rather than ``HydraRunner().run`` directly) so
    that ``@hydra.main`` populates ``HydraConfig`` — ``HydraRunner.derive_run_name``
    reads ``HydraConfig.get().overrides.task`` to append the sweep suffix to the
    run name.

    Args:
        config_path: Absolute path to the training config yaml.
        overrides: Hydra-style overrides for this sweep combination.
        cwd: Working directory at submission time. Restored so user code referenced
            via ``config_schema_kind`` is importable.
    """
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    os.chdir(cwd)

    sys.argv = [sys.argv[0], "--hp", config_path, *overrides]

    # Imported lazily: ``main_train`` calls ``setup_hydra()`` at import time, which
    # inspects ``sys.argv`` — so argv must be set up first.
    from noether.training.cli.main_train import main as train_main

    train_main()


def _format_task_preview(config_path: Path, combo: list[str]) -> str:
    """Render a one-line preview of the (in-process) training invocation."""
    return f"noether-train --hp {config_path} {' '.join(combo)}".rstrip()


def _print_dry_run(folder: str, params: dict[str, Any], previews: list[str]) -> None:
    print("[dry-run] submitit AutoExecutor configuration:")
    print(f"  folder: {folder}")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print(f"[dry-run] Would submit {len(previews)} task(s) in-process via main_train.main:")
    for idx, preview in enumerate(previews):
        print(f"  [{idx}] {preview}")


def _submit(
    executor: submitit.AutoExecutor,
    config_path: Path,
    combos: list[list[str]],
    cwd: str,
) -> list[submitit.Job]:
    """Submit one or more in-process training tasks.

    Multiple combos go inside ``executor.batch()`` so SLURM sees a single array job.
    """
    if len(combos) == 1:
        return [executor.submit(_train_entrypoint, str(config_path), combos[0], cwd)]
    with executor.batch():
        return [executor.submit(_train_entrypoint, str(config_path), combo, cwd) for combo in combos]


def main() -> None:
    """Entry point for ``noether-train-submit-job``.

    Validates a Hydra config (and every multirun combination thereof) and submits
    a training job — or a SLURM array job — via :mod:`submitit`.
    """
    if "--help" in sys.argv or "-h" in sys.argv:
        print(_HELP_TEXT)
        sys.exit(0)

    # Make user code importable for schemas referenced via ``config_schema_kind``.
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())

    raw_config_path, overrides, multirun, dry_run = _parse_argv(sys.argv)
    config_path = Path(raw_config_path).resolve()
    if not config_path.exists():
        print(f"Error: config file does not exist: {config_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Config path: {config_path}")
    print(f"Multirun: {multirun}")

    try:
        combos = _expand_sweeps(overrides, multirun)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    slurm_config = _validate_all_combos(config_path, combos)
    folder, params = slurm_config.to_executor_kwargs()
    cwd = os.getcwd()

    if dry_run:
        previews = [_format_task_preview(config_path, combo) for combo in combos]
        _print_dry_run(folder, params, previews)
        sys.exit(0)

    executor = submitit.AutoExecutor(folder=folder)
    executor.update_parameters(**params)
    jobs = _submit(executor, config_path, combos, cwd)

    if len(jobs) == 1:
        print(f"Submitted job {jobs[0].job_id}")
    else:
        print(f"Submitted SLURM array of {len(jobs)} tasks (master id {jobs[0].job_id.split('_')[0]}):")
        for job in jobs:
            print(f"  {job.job_id}")


if __name__ == "__main__":
    main()
