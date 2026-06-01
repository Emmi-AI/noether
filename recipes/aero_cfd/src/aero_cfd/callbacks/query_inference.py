#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

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

    The first chunk runs a full forward pass against the AB-UPT backbone and
    keeps the returned ``kv_cache`` (geometry encoding + RoPE, anchor K/V in
    every physics and decoder block). Subsequent chunks reuse that cache and
    pass only the new query slice, so the geometry encoder + anchor projection
    run exactly once per sample. Anchor predictions come from the first chunk
    only — subsequent chunks return query outputs only.
    """

    def __init__(self, callback_config: QueryInferenceCallbackConfig, **kwargs):
        super().__init__(callback_config, **kwargs)
        self.num_surface_anchors = callback_config.num_surface_anchors
        self.num_volume_anchors = callback_config.num_volume_anchors
        self.query_chunk_size = callback_config.query_chunk_size

    def _run_model_inference(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run chunked query-based inference with geometry/anchor KV caching.

        Splits batch positions into anchors + queries, runs the first chunk
        through the full backbone, then iterates remaining chunks reusing the
        first chunk's cache. Returns combined anchor + query outputs.
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

        cs = self.query_chunk_size
        n_chunks = max(1, math.ceil(n_sq / cs), math.ceil(n_vq / cs))

        # Bypass the AeroABUPT wrapper (which discards the kv_cache) and drive
        # the AnchoredBranchedUPT backbone directly so the cache survives
        # across chunks.
        backbone = self.model.backbone
        anchor_positions = {"surface": surface_anchors, "volume": volume_anchors}

        anchor_outputs: dict[str, torch.Tensor] = {}
        query_chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
        kv_cache: dict[str, Any] | None = None

        for i in range(n_chunks):
            s_start, s_end = i * cs, min((i + 1) * cs, n_sq)
            v_start, v_end = i * cs, min((i + 1) * cs, n_vq)

            query_positions: dict[str, torch.Tensor] = {}
            if s_start < n_sq:
                query_positions["surface"] = surface_queries[:, s_start:s_end]
            if v_start < n_vq:
                query_positions["volume"] = volume_queries[:, v_start:v_end]

            with self.trainer.autocast_context:
                if i == 0:
                    out, kv_cache = backbone(
                        geometry_position=batch["geometry_position"],
                        geometry_supernode_idx=batch["geometry_supernode_idx"],
                        geometry_batch_idx=batch["geometry_batch_idx"],
                        domain_anchor_positions=anchor_positions,
                        domain_query_positions=query_positions or None,
                    )
                else:
                    # Reuse cached geometry encoding + anchor K/V; the backbone
                    # refuses geometry tensors / anchor positions when
                    # 'physics_blocks' is in the cache, and accepts a subset of
                    # domains in query_positions when others are exhausted.
                    out, _ = backbone(
                        domain_query_positions=query_positions,
                        kv_cache=kv_cache,
                    )

            for key, value in out.items():
                if key.startswith("query_"):
                    base_key = key[len("query_") :]
                    query_chunks[base_key].append(value)
                elif i == 0:
                    # Anchor outputs come from chunk 0 only. With anchors_cached
                    # the backbone emits query_* keys exclusively.
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
