#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Pipeline for the wrapped latent-diffusion dataset.

Reuses the AB-UPT sample processors and collators from
:class:`AeroMultistagePipeline` (so anchor positions, geometry tokens, and
field targets are produced exactly the same way as during AE training)
and appends a small :class:`DefaultCollator` that batches the latent-side
items written by :func:`extract_latents` (``latents``,
``supernode_positions``, optional per-branch ``super_position_*``).
"""

from __future__ import annotations

from aero_cfd.pipeline.multistage_pipelines.aero_multistage import (
    AeroCFDPipelineConfig,
    AeroMultistagePipeline,
)

from noether.data.pipeline.collators import DefaultCollator


class LatentDiffusionPipelineConfig(AeroCFDPipelineConfig):
    """Pipeline config for :class:`LatentDataset`.

    Inherits every aero-CFD sample-processing knob (anchor counts, geometry
    sampling, etc.) so the pipeline produces the same per-sample tensors as
    the AB-UPT autoencoder training pipeline.
    """

    kind: str | None = "datasets.latent_diffusion_pipeline.LatentDiffusionPipeline"


class LatentDiffusionPipeline(AeroMultistagePipeline):
    """AB-UPT pipeline + latent collator.

    The base class processes the raw CFD positions/fields into the keys the
    diffusion eval consumes (``surface_anchor_position``,
    ``volume_anchor_position``, ``geometry_position``,
    ``geometry_supernode_idx``, ``geometry_batch_idx``, plus the
    ``{domain}_{field}_target`` keys). This subclass tacks on a final
    collator that stacks the latent-side items so they show up in the batch
    too — without it ``DefaultCollator``'s explicit ``items`` list would
    silently drop them.
    """

    def _build_collator_pipeline(self) -> list:
        collators = super()._build_collator_pipeline()
        collators.append(
            DefaultCollator(
                items=["latents", "supernode_positions"],
                optional_items=["super_position_surface", "super_position_volume"],
            )
        )
        return collators
