#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import math
from collections import defaultdict

import torch

from .aero_metrics import AeroMetricsCallback, AeroMetricsCallbackConfig


class QueryInferenceCallbackConfig(AeroMetricsCallbackConfig):
    """Configuration for query-based dense inference.

    Extends :class:`AeroMetricsCallbackConfig` with parameters that control
    how the batch positions are split into training-sized anchors and additional
    query points, and how query chunks are processed.
    """

    kind: str | None = "aero_cfd.callbacks.QueryInferenceCallback"

    num_surface_anchors: int
    """Number of surface positions to treat as anchors (must match training)."""
    num_volume_anchors: int
    """Number of volume positions to treat as anchors (must match training)."""
    query_chunk_size: int = 10000
    """Max query points per domain per forward pass."""


class QueryInferenceCallback(AeroMetricsCallback):
    """Evaluation callback that performs chunked query-based inference.

    The inference dataset produces more surface/volume positions than training.
    This callback splits them into:
    - **Anchors** (first ``num_*_anchors`` positions) — same as training
    - **Queries** (the rest) — processed in chunks

    Each forward pass receives fixed anchors + one chunk of queries.  The model
    outputs anchor predictions (constant across chunks) and query predictions
    (concatenated).  Metrics and VTK export use the combined result.
    """

    def __init__(self, callback_config: QueryInferenceCallbackConfig, **kwargs):
        super().__init__(callback_config, **kwargs)
        self.num_surface_anchors = callback_config.num_surface_anchors
        self.num_volume_anchors = callback_config.num_volume_anchors
        self.query_chunk_size = callback_config.query_chunk_size

    def _run_model_inference(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run chunked query-based inference.

        Splits batch positions into anchors + queries, iterates through query
        chunks while keeping anchors fixed, and returns combined outputs.
        """
        # Split positions: [anchors | queries]
        surface_all = batch["surface_anchor_position"]
        volume_all = batch["volume_anchor_position"]

        surface_anchors = surface_all[:, : self.num_surface_anchors]
        surface_queries = surface_all[:, self.num_surface_anchors :]
        volume_anchors = volume_all[:, : self.num_volume_anchors]
        volume_queries = volume_all[:, self.num_volume_anchors :]

        n_sq = surface_queries.shape[1]
        n_vq = volume_queries.shape[1]

        if n_sq == 0 and n_vq == 0:
            # No queries — standard anchor-only inference
            return super()._run_model_inference(batch)

        base_kwargs = {
            "geometry_position": batch["geometry_position"],
            "geometry_supernode_idx": batch["geometry_supernode_idx"],
            "geometry_batch_idx": batch["geometry_batch_idx"],
            "surface_anchor_position": surface_anchors,
            "volume_anchor_position": volume_anchors,
        }

        cs = self.query_chunk_size
        n_chunks = max(1, math.ceil(n_sq / cs), math.ceil(n_vq / cs))

        anchor_outputs: dict[str, torch.Tensor] = {}
        query_chunks: dict[str, list[torch.Tensor]] = defaultdict(list)

        for i in range(n_chunks):
            chunk_kwargs = dict(base_kwargs)

            s_start, s_end = i * cs, min((i + 1) * cs, n_sq)
            v_start, v_end = i * cs, min((i + 1) * cs, n_vq)

            if s_start < n_sq:
                chunk_kwargs["query_surface_position"] = surface_queries[:, s_start:s_end]
            if v_start < n_vq:
                chunk_kwargs["query_volume_position"] = volume_queries[:, v_start:v_end]

            with self.trainer.autocast_context:
                out = self.model(**chunk_kwargs)

            for key, value in out.items():
                if key.startswith("query_"):
                    base_key = key[len("query_") :]
                    query_chunks[base_key].append(value)
                elif i == 0:
                    # Anchor outputs vary slightly across chunks due to decoder
                    # self-attention over [anchors, queries]. Keep first chunk's.
                    anchor_outputs[key] = value

        # Combine: [anchor_predictions, query_predictions]
        combined: dict[str, torch.Tensor] = {}
        for key, anchor_val in anchor_outputs.items():
            chunks = query_chunks.get(key, [])
            if chunks:
                combined[key] = torch.cat([anchor_val] + chunks, dim=1)
            else:
                combined[key] = anchor_val

        return combined
