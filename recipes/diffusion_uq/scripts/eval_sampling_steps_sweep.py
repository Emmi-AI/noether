#!/usr/bin/env python3
#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Sweep ``sampling_steps`` for a trained data-space AB-UPT diffusion run.

Single in-process eval pass: builds one
:class:`DataspaceDiffusionChunkedEvalCallback` configured with every requested
step count, and dispatches via :func:`noether.inference.evaluate`. The callback
emits each step's metrics under a ``steps_{n}/`` prefix, so a single wandb run
ends up with every step count side-by-side. Single GPU only (pin via
``--device-id``).

Example::

    uv run python -m scripts.eval_sampling_steps_sweep \\
        --run-dir outputs/abupt_diffusion/30035_2026-05-11_spk1e \\
        --steps 1 2 4 8 16 \\
        --resume-checkpoint best_model.loss.test.total
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from callbacks.dataspace_diffusion_chunked_eval import DataspaceDiffusionChunkedEvalCallbackConfig
from experiments import ABUPT_FORWARD_PROPERTIES

from noether.inference import evaluate
from noether.modeling.diffusion.flow_matching import FlowMatchingConfig

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep sampling_steps over eval runs")
    p.add_argument(
        "--run-dir",
        type=str,
        default="outputs/abupt_diffusion/30035_2026-05-11_spk1e",
        help="Training run output directory containing hp_resolved.yaml",
    )
    p.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="sampling_steps values to evaluate in a single pass",
    )
    p.add_argument(
        "--resume-checkpoint",
        type=str,
        default="best_model.loss.test.total",
        help="Checkpoint tag (best_model.loss.test.total / latest / E100 / U2500 / S40000)",
    )
    p.add_argument(
        "--stage-name",
        type=str,
        default="eval_steps_sweep",
        help="Stage name for this eval run's outputs (logs / wandb / saved metrics)",
    )
    p.add_argument(
        "--dataset-key",
        type=str,
        default="test",
        help="Dataset key to evaluate against",
    )
    p.add_argument(
        "--device-id",
        type=str,
        default="0",
        help="Physical GPU id to pin via CUDA_VISIBLE_DEVICES",
    )
    p.add_argument(
        "--disable-tracker",
        action="store_true",
        help="Disable wandb tracker.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    run_dir = Path(args.run_dir)

    logger.info(f"sweeping sampling_steps over {args.steps} in a single eval pass")

    eval_callback = DataspaceDiffusionChunkedEvalCallbackConfig(
        kind="callbacks.dataspace_diffusion_chunked_eval.DataspaceDiffusionChunkedEvalCallback",
        every_n_epochs=1,
        dataset_key=args.dataset_key,
        forward_properties=ABUPT_FORWARD_PROPERTIES,
        chunked_inference=False,
        sampling_steps=args.steps,
        schedule_config=FlowMatchingConfig(),
    )

    evaluate(
        run_dir=run_dir,
        resume_checkpoint=args.resume_checkpoint,
        stage_name=args.stage_name,
        callbacks=[eval_callback],
        device="cuda",
        disable_tracker=args.disable_tracker,
    )

    logger.info("sweep complete")


if __name__ == "__main__":
    main()
