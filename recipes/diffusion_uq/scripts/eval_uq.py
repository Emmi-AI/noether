#!/usr/bin/env python3
#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Run :class:`DataspaceDiffusionUQCallback` against a trained AB-UPT diffusion run.

Single in-process eval pass that draws ``n_uq_samples`` FM samples per
geometry and reports per-point std-vs-|error| correlation plus Cd / Cl mean /
std / empirical CI calibration. Figures are written to
``<run_dir>/<stage_name>/uq/<dataset_key>/`` as PNGs.

Requires ``surface_normals`` / ``surface_area`` / ``surface_position`` to be
present in the batch (they're needed for force-coefficient integration).

Example::

    uv run python -m scripts.eval_uq \\
        --run-dir outputs/abupt_diffusion/30035_2026-05-11_spk1e \\
        --n-uq-samples 10 \\
        --sampling-steps 10 \\
        --resume-checkpoint best_model.loss.test.total
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from callbacks.dataspace_diffusion_uq import DataspaceDiffusionUQCallbackConfig
from experiments import ABUPT_FORWARD_PROPERTIES

from noether.core.schemas.dataset import SubsetWrapperConfig
from noether.core.schemas.schema import ConfigSchema
from noether.inference import evaluate
from noether.modeling.diffusion.flow_matching import FlowMatchingConfig

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the diffusion UQ eval callback")
    p.add_argument(
        "--run-dir",
        type=str,
        default="outputs/abupt_diffusion/30035_2026-05-11_spk1e",
        help="Training run output directory containing hp_resolved.yaml",
    )
    p.add_argument(
        "--n-uq-samples",
        type=int,
        default=6,
        help="Number of independent FM draws per geometry (>=2). Cost scales linearly.",
    )
    p.add_argument(
        "--sampling-steps",
        type=int,
        default=4,
        help="FM Euler step count per draw",
    )
    p.add_argument(
        "--scatter-max-points-per-sample",
        type=int,
        default=2000,
        help="Per-geometry cap on points contributed to the pooled std-vs-|error| scatter",
    )
    p.add_argument(
        "--resume-checkpoint",
        type=str,
        default="latest",
        help="Checkpoint tag (best_model.loss.test.total / latest / E100 / U2500 / S40000)",
    )
    p.add_argument(
        "--stage-name",
        type=str,
        default="eval_uq",
        help="Stage name for this eval run's outputs (logs / wandb / saved metrics / figures)",
    )
    p.add_argument(
        "--dataset-key",
        type=str,
        default="chunked_test",
        help="Dataset key to evaluate against",
    )
    p.add_argument(
        "--device-id",
        type=str,
        default="0",
        help="Physical GPU id to pin via CUDA_VISIBLE_DEVICES",
    )
    p.add_argument(
        "--num-surface-anchors",
        type=int,
        default=16384,
        help="Number of surface positions to treat as anchors (must match training).",
    )
    p.add_argument(
        "--num-volume-anchors",
        type=int,
        default=16384,
        help="Number of volume positions to treat as anchors (must match training).",
    )
    p.add_argument(
        "--query-chunk-size",
        type=int,
        default=16384 * 2,
        help="Max query points per domain per forward pass within a single Euler step.",
    )
    p.add_argument(
        "--num-test-samples",
        type=int,
        default=10,
        help="Number of geometries to evaluate on (for quick testing; set via SubsetWrapper in config_overrides)",
    )
    p.add_argument(
        "--disable-tracker",
        action="store_true",
        help="Disable wandb tracker.",
    )
    p.add_argument(
        "--stl-root-path",
        type=str,
        default=None,
        help="Optional root directory holding the raw DrivAerML STL files",
    )
    return p.parse_args(argv)


def _enable_force_inputs(config: ConfigSchema, num_query_points: int, num_test_samples: int) -> None:
    """Toggle the dataset settings the training config baked out so the UQ
    callback's Cd / Cl integration finds its inputs in the batch.

    Training defaults exclude ``surface_normals`` / ``surface_area`` and leave
    ``use_surface_position_as_input=False``, so without this the eval batch is
    missing all three mesh tensors and force computation silently no-ops.
    """
    config.trainer.effective_batch_size = 1
    for ds_config in config.datasets.values():
        if ds_config.excluded_properties:
            ds_config.excluded_properties -= {"surface_normals", "surface_area"}
        ds_config.pipeline.use_surface_position_as_input = True
        ds_config.pipeline.num_surface_anchor_points = num_query_points
        ds_config.pipeline.num_volume_anchor_points = num_query_points
        ds_config.dataset_wrappers = [SubsetWrapperConfig(start_index=0, end_index=num_test_samples)]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    run_dir = Path(args.run_dir)

    logger.info(
        f"running UQ eval: n_uq_samples={args.n_uq_samples}, "
        f"sampling_steps={args.sampling_steps}, dataset={args.dataset_key!r}"
    )

    uq_callback = DataspaceDiffusionUQCallbackConfig(
        kind="callbacks.dataspace_diffusion_uq.DataspaceDiffusionUQCallback",
        every_n_epochs=1,
        dataset_key=args.dataset_key,
        forward_properties=ABUPT_FORWARD_PROPERTIES,
        compute_forces=True,
        sampling_steps=[args.sampling_steps],
        n_uq_samples=args.n_uq_samples,
        scatter_max_points_per_sample=args.scatter_max_points_per_sample,
        schedule_config=FlowMatchingConfig(),
        num_surface_anchors=args.num_surface_anchors,
        num_volume_anchors=args.num_volume_anchors,
        query_chunk_size=args.query_chunk_size,
        stl_root_path=args.stl_root_path,
    )

    evaluate(
        run_dir=run_dir,
        resume_checkpoint=args.resume_checkpoint,
        stage_name=args.stage_name,
        callbacks=[uq_callback],
        config_overrides=lambda config: _enable_force_inputs(config, 1_000_000, args.num_test_samples),
        device="cuda",
        disable_tracker=args.disable_tracker,
    )

    logger.info("UQ eval complete")


if __name__ == "__main__":
    main()
