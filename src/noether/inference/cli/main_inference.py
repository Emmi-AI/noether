#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
import os
import sys
from pathlib import Path

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from noether.inference.run import sanitize_hp_resolved
from noether.inference.runners.inference_runner import InferenceRunner
from noether.training.cli import setup_hydra

logger = logging.getLogger(__name__)


_LEGACY_NAV_KEYS = ("input_dir", "run_id", "stage_name")


def _pop_eval_path_args(argv: list[str]) -> tuple[dict[str, str], list[str]]:
    """Extract path-navigation args from ``argv``.

    Always pops ``run_dir=...`` (and ``+run_dir=...``). The legacy trio ``input_dir``/``run_id``/``stage_name`` is only
    popped when ``input_dir`` is supplied — otherwise plain ``run_id=foo`` is left alone so it can act as a normal
    config override on top of the loaded training config.
    """
    has_input_dir = any(a.startswith(("input_dir=", "+input_dir=")) for a in argv)
    keys_to_pop = {"run_dir", *(_LEGACY_NAV_KEYS if has_input_dir else ())}

    popped: dict[str, str] = {}
    remaining: list[str] = []
    for arg in argv:
        for key in keys_to_pop:
            if arg.startswith((f"{key}=", f"+{key}=")):
                _, value = arg.split("=", 1)
                popped[key] = value
                break
        else:
            remaining.append(arg)
    return popped, remaining


def _build_run_dir(path_values: dict[str, str]) -> Path | None:
    """Pick the user-supplied run output directory and return it as an absolute path.

    ``path_values`` is the dict produced by :func:`_pop_eval_path_args` — it
    holds whichever of ``run_dir`` / ``input_dir`` / ``run_id`` / ``stage_name``
    the user passed on the command line. This function chooses *which* of those
    inputs to use (preferring the canonical ``run_dir``, falling back to the
    deprecated three-part form) and assembles the final path; the returned
    value is purely the user's choice, normalized to an absolute path via
    :py:meth:`pathlib.Path.resolve`.

    Resolution order:

    1. If ``run_dir`` was supplied, use it directly.
    2. Else, if both ``input_dir`` and ``run_id`` were supplied, reconstruct
       ``input_dir/run_id[/stage_name]`` (logs a deprecation warning).
    3. Otherwise return ``None`` so the caller can defer to Hydra/the runner
       to surface a clearer error.
    """
    if "run_dir" in path_values:
        return Path(path_values["run_dir"]).resolve()
    if "input_dir" in path_values and "run_id" in path_values:
        logger.warning(
            "input_dir/run_id/stage_name is deprecated; pass `run_dir=<path>` instead "
            "(the directory that contains hp_resolved.yaml)."
        )
        return (Path(path_values["input_dir"]) / path_values["run_id"] / path_values.get("stage_name", "")).resolve()
    return None


def _inject_hp_resolved_into_argv() -> None:
    """Pre-process ``sys.argv`` so Hydra loads ``run_dir/hp_resolved.yaml`` as
    the base config.

    This lets users override any training-time key (e.g. ``tracker=disabled``,
    ``trainer.max_epochs=1``) without the Hydra ``+`` force-add prefix, the
    same way ``noether-train --hp <config>.yaml`` works.

    A user-supplied ``--hp <other.yaml>`` takes precedence and is left alone
    (escape hatch for power users composing their own eval config).
    """
    if len(sys.argv) < 2:
        return
    if "--help" in sys.argv or "-h" in sys.argv:
        return
    if "--hp" in sys.argv:
        return

    path_values, remaining = _pop_eval_path_args(sys.argv[1:])
    run_dir = _build_run_dir(path_values)
    if run_dir is None:
        return  # let setup_hydra/main raise a clearer error

    hp_resolved = run_dir / "hp_resolved.yaml"
    if not hp_resolved.exists():
        raise FileNotFoundError(
            f"hp_resolved.yaml not found in {run_dir}. "
            "Make sure run_dir points at a training run output directory "
            "(typically output_path/run_id[/stage_name])."
        )
    safe_hp = sanitize_hp_resolved(hp_resolved)

    # `hp_resolved.yaml` is dumped with `exclude_unset=True`, so values that were generated at training-time
    # (e.g. `run_id`) are absent. Infer them from the run_dir path and inject as forced overrides so the eval run
    # writes alongside the training run and resume picks the right checkpoint.
    with open(safe_hp) as f:
        hp_data = yaml.safe_load(f)
    stage_name = hp_data.get("stage_name") or ""
    inferred_run_id = run_dir.parent.name if stage_name else run_dir.name
    run_id = hp_data.get("run_id") or inferred_run_id

    injected = [
        f"++run_id='{run_id}'",
        f"++stage_name={stage_name}",
        f"++resume_run_id='{run_id}'",
        f"++resume_stage_name={stage_name}",
        # Default to the latest checkpoint; user-supplied `resume_checkpoint=...` in `remaining` is applied afterwards
        # and wins.
        "++resume_checkpoint=latest",
    ]
    # If the source recorded its training-time ``output_path``, pin it as the resume source root. That decouples
    # checkpoint lookup from whatever ``output_path`` the user picks for this eval run (e.g. redirecting eval
    # outputs to ``/scratch/...`` no longer breaks resume).
    source_output_path = hp_data.get("output_path")
    if source_output_path:
        injected.append(f"++resume_output_path={source_output_path}")
    sys.argv = [sys.argv[0], "--hp", safe_hp.as_posix(), *injected, *remaining]


_inject_hp_resolved_into_argv()
setup_hydra()


@hydra.main(config_path=None, config_name=None, version_base="1.3")
def main(eval_config: DictConfig) -> None:
    """Entry point for ``noether-eval``.

    The training run's ``hp_resolved.yaml`` has already been wired in as the Hydra base config (see
    :func:`_inject_hp_resolved_into_argv`), so this function only needs to set the resume fields and dispatch.

    Examples:
        noether-eval run_dir=outputs/2026-01-10_abc12/train
        noether-eval run_dir=outputs/2026-01-10_abc12/train resume_checkpoint=best
        noether-eval run_dir=outputs/2026-01-10_abc12/train tracker=disabled trainer.max_epochs=1
    """
    # disable hydra changing working directory and add cwd to PYTHONPATH
    os.chdir(hydra.utils.get_original_cwd())
    sys.path.insert(0, hydra.utils.get_original_cwd())

    if eval_config.get("resume_run_id") is None:
        raise ValueError(
            "noether-eval requires `run_dir=<path>` (the training run output "
            "directory containing hp_resolved.yaml). "
            "Example: noether-eval run_dir=outputs/2026-01-10_abc12/train"
        )

    config_dict = OmegaConf.to_container(eval_config, resolve=True)
    InferenceRunner().run(config_dict)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
