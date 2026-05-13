#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Programmatic eval API — Python-side equivalent of ``noether-eval``.

The CLI does its work in :mod:`noether.inference.cli.main_inference`: it
loads ``<run_dir>/hp_resolved.yaml`` as the Hydra base config, injects
``resume_*`` overrides, and dispatches through :class:`InferenceRunner`. This
module exposes the same flow as a normal function so Python callers (e.g.
notebooks, sweep scripts) don't have to shell out.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from noether.core.configs.hyperparameters import Hyperparameters
from noether.core.schemas.callbacks import CallBackBaseConfig
from noether.core.schemas.schema import ConfigSchema
from noether.inference.runners.inference_runner import InferenceRunner

logger = logging.getLogger(__name__)


def _set_resume_from_run_dir(config: ConfigSchema, run_dir: Path) -> None:
    """Wire ``resume_*`` and ``run_id``/``stage_name`` so eval reads the
    training checkpoint and writes alongside the source run.

    Mirrors :func:`noether.inference.cli.main_inference._inject_hp_resolved_into_argv`:
    if the saved config carries an explicit ``run_id`` use it; otherwise infer
    it from ``run_dir``'s name (or its parent's, if ``stage_name`` is set).
    """
    saved_stage_name = config.stage_name or ""
    inferred_run_id = run_dir.parent.name if saved_stage_name else run_dir.name
    source_run_id = config.run_id or inferred_run_id

    config.run_id = source_run_id
    config.stage_name = saved_stage_name
    config.resume_run_id = source_run_id
    config.resume_stage_name = saved_stage_name
    if config.output_path:
        # Pin the source root so a later ``output_path`` override doesn't
        # redirect checkpoint lookup away from the training run.
        config.resume_output_path = Path(config.output_path)


def evaluate(
    run_dir: str | Path,
    *,
    resume_checkpoint: str = "latest",
    stage_name: str | None = None,
    callbacks: list[CallBackBaseConfig] | None = None,
    device: str = "cuda",
    disable_tracker: bool = False,
) -> None:
    """Run evaluation against a training run directory.

    Programmatic equivalent of::

        noether-eval run_dir=<run_dir> resume_checkpoint=<...> ...

    Loads ``<run_dir>/hp_resolved.yaml`` via :meth:`Hyperparameters.load_resolved`,
    wires the ``resume_*`` fields so checkpoints are read from the training
    run, optionally replaces the trainer callback list, and dispatches through
    :meth:`InferenceRunner.main` (single-process, no Hydra/CLI involvement).

    Args:
        run_dir: Training run output directory — the one that contains
            ``hp_resolved.yaml``. Typically ``<output_path>/<run_id>[/<stage_name>]``.
        resume_checkpoint: Checkpoint tag to load. Examples: ``"latest"``,
            ``"best_model.<metric>"``, ``"E100"`` (epoch 100), ``"U2500"``
            (update 2500), ``"S40000"`` (sample 40000).
        stage_name: Optional sub-stage name for *this* eval run's outputs.
            Logs / wandb / saved metrics land under
            ``<run_dir>/<stage_name>/``, separate from the training outputs.
            Leave ``None`` to write alongside the training run.
        callbacks: If provided, replaces ``config.trainer.callbacks`` for the
            eval run. Pass the exact callbacks that should execute (e.g. a
            single sampling/rollout callback) — nothing from the training
            config's callback list is kept.
        device: Device string passed to the trainer (default ``"cuda"``).
            For multi-GPU eval use the ``noether-eval`` CLI; this function
            is single-process.
        disable_tracker: If ``True``, drop the saved tracker config so eval
            doesn't create a new wandb run.

    Raises:
        FileNotFoundError: if ``run_dir`` doesn't contain ``hp_resolved.yaml``.

    Example::

        from noether.inference import evaluate
        from my_recipe.callbacks import SamplingCallbackConfig

        for steps in [1, 2, 4, 8, 16]:
            evaluate(
                run_dir="outputs/abupt_diffusion/30035_2026-05-11_spk1e",
                resume_checkpoint="best_model.loss.test.total",
                stage_name=f"eval_steps{steps:02d}",
                callbacks=[SamplingCallbackConfig(every_n_epochs=1, sampling_steps=steps)],
            )
    """
    run_dir = Path(run_dir).resolve()
    hp_path = run_dir / "hp_resolved.yaml"
    if not hp_path.exists():
        raise FileNotFoundError(
            f"hp_resolved.yaml not found in {run_dir}. run_dir must point at a training run output directory."
        )

    # Recipe-style runs (kind paths like ``models.foo.Bar``) need the recipe
    # root on sys.path so discriminated-config dynamic imports resolve. The
    # CLI does ``sys.path.insert(0, hydra.utils.get_original_cwd())``; here we
    # add the current working directory for the same effect — callers that
    # ``cd`` into the recipe before calling ``evaluate`` get it for free.
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    config = Hyperparameters.load_resolved(hp_path)
    _set_resume_from_run_dir(config, run_dir)
    config.resume_checkpoint = resume_checkpoint

    if stage_name is not None:
        config.stage_name = stage_name
    if disable_tracker:
        config.tracker = None
    if callbacks is not None:
        config.trainer.callbacks = callbacks

    logger.info(
        f"evaluating run_id={config.run_id!r} stage_name={config.stage_name!r} resume_checkpoint={resume_checkpoint!r}"
    )
    InferenceRunner.main(device=device, config=config)
