#!/usr/bin/env python3
#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Run :class:`aero_cfd.callbacks.QueryInferenceCallback` against a trained AB-UPT
regression run.

The regression model is trained with a fixed number of surface/volume anchors
(``num_*_anchor_points`` in the training pipeline). The ``chunked_test`` dataset
in the resumed run is configured to return *all* mesh points (1e9 anchor cap),
so the callback splits each batch into ``[anchors | queries]`` and runs the
model over query chunks with anchors fixed, producing dense predictions on the
full mesh.

Example::

    uv run python -m scripts.eval_query_inference \\
        --run-dir outputs/abupt_diffusion/30403_2026-05-13_wysns \\
        --resume-checkpoint best_model.loss.test.total
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path, PosixPath

import yaml
from aero_cfd.callbacks import QueryInferenceCallbackConfig
from experiments import ABUPT_REGRESSION_FORWARD_PROPERTIES

from noether.core.initializers.previous_run import PreviousRunInitializer
from noether.core.schemas.dataset import SubsetWrapperConfig
from noether.core.schemas.schema import ConfigSchema
from noether.inference import evaluate

logger = logging.getLogger(__name__)


# hp_resolved.yaml saved under Python 3.13 tags PosixPath as
# ``pathlib._local.PosixPath`` (the internal module), which yaml.FullLoader
# refuses by default. Teach it that this tag is just a PosixPath.
def _construct_posix_path(loader: yaml.FullLoader, node: yaml.Node) -> PosixPath:
    args = loader.construct_sequence(node)
    return PosixPath(*args)


yaml.FullLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:pathlib._local.PosixPath",
    _construct_posix_path,
)


# Checkpoint compat: this run was trained on 2026-05-13, before commit 46b63366
# (2026-05-18) which replaced ``nn.LayerNorm`` with ``nn.RMSNorm`` for the
# AB-UPT readout. RMSNorm has no ``.bias``, so loading the saved
# ``*.norm_final.bias`` keys fails with strict load_state_dict. Inject a
# ``patterns_to_remove`` filter on the PreviousRunInitializer that drops them.
_orig_init = PreviousRunInitializer.__init__


def _patched_init(self, initializer_config, **kwargs):  # noqa: ANN001, ANN202
    _orig_init(self, initializer_config, **kwargs)
    if "norm_final.bias" not in self.patterns_to_remove:
        self.patterns_to_remove.append("norm_final.bias")


PreviousRunInitializer.__init__ = _patched_init


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the QueryInferenceCallback eval")
    p.add_argument(
        "--run-dir",
        type=str,
        default="outputs/abupt_diffusion/30403_2026-05-13_wysns",
        help="Training run output directory containing hp_resolved.yaml",
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
        default="eval_query_inference",
        help="Stage name for this eval run's outputs (logs / wandb / saved metrics)",
    )
    p.add_argument(
        "--dataset-key",
        type=str,
        default="chunked_test",
        help="Dataset key to evaluate against (chunked_test returns the full mesh)",
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
        help="Max query points per domain per forward pass.",
    )
    p.add_argument(
        "--num-test-samples",
        type=int,
        default=None,
        help="Optional cap on number of geometries to evaluate (SubsetWrapper). Default: all.",
    )
    p.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save per-sample denormalized predictions to <run_dir>/<stage_name>/predictions.",
    )
    p.add_argument(
        "--compute-forces",
        action="store_true",
        help="Compute Cd / Cl errors (requires surface_normals / surface_area / surface_position).",
    )
    p.add_argument(
        "--precision",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Inference precision. Defaults to float16 (matches training) for ~10x faster attention.",
    )
    p.add_argument(
        "--disable-tracker",
        action="store_true",
        help="Disable wandb tracker.",
    )
    return p.parse_args(argv)


def _configure_datasets(
    config: ConfigSchema,
    *,
    compute_forces: bool,
    num_test_samples: int | None,
    precision: str,
) -> None:
    """Set batch size + precision, and (optionally) cap samples / enable force inputs."""
    config.trainer.effective_batch_size = 1
    config.trainer.precision = precision

    for ds_config in config.datasets.values():
        if compute_forces:
            if ds_config.excluded_properties:
                ds_config.excluded_properties -= {"surface_normals", "surface_area"}
            ds_config.pipeline.use_surface_position_as_input = True
        if num_test_samples is not None:
            ds_config.dataset_wrappers = [SubsetWrapperConfig(start_index=0, end_index=num_test_samples)]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    run_dir = Path(args.run_dir)
    predictions_path: str | None = None
    if args.save_predictions:
        stage_dir = run_dir / args.stage_name if args.stage_name else run_dir
        predictions_path = str(stage_dir / "predictions")

    logger.info(
        f"running query-inference eval on {args.dataset_key!r}: "
        f"anchors=({args.num_surface_anchors}, {args.num_volume_anchors}), "
        f"chunk={args.query_chunk_size}"
    )

    callback = QueryInferenceCallbackConfig(
        every_n_epochs=1,
        dataset_key=args.dataset_key,
        forward_properties=ABUPT_REGRESSION_FORWARD_PROPERTIES,
        num_surface_anchors=args.num_surface_anchors,
        num_volume_anchors=args.num_volume_anchors,
        query_chunk_size=args.query_chunk_size,
        save_predictions=args.save_predictions,
        predictions_path=predictions_path,
        batch_properties_to_save=["surface_anchor_position", "volume_anchor_position"],
        compute_forces=args.compute_forces,
    )

    evaluate(
        run_dir=run_dir,
        resume_checkpoint=args.resume_checkpoint,
        stage_name=args.stage_name,
        callbacks=[callback],
        config_overrides=lambda config: _configure_datasets(
            config,
            compute_forces=args.compute_forces,
            num_test_samples=args.num_test_samples,
            precision=args.precision,
        ),
        device="cuda",
        disable_tracker=args.disable_tracker,
    )

    logger.info("query-inference eval complete")


if __name__ == "__main__":
    main()
