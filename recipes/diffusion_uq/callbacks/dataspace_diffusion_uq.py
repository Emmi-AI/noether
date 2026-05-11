#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Uncertainty-quantification eval callback for AB-UPT data-space diffusion.

Subclasses the chunked schedule-driven eval callback. Instead of one
denoising pass per test sample, runs ``n_uq_samples`` independent draws and
turns the resulting ensemble into:

* Per-field per-point Pearson correlation between predictive std and absolute
  error — the quantity plotted in
  ``03_dataspace_diffusion.ipynb`` (`plot_field_uq_stl`).
* Drag and lift coefficient mean / std / empirical 95% CI per geometry,
  aggregated into R², coverage@1σ, and mean ``|z|`` calibration metrics
  against GT Cd / Cl (mirrors the integrated-quantity UQ cell of the
  notebook).

Logs scalars under ``loss/<dataset>/...`` and saves matplotlib figures
(calibration scatter, std-vs-error scatter) both to disk and via
``writer.add_nonscalar`` so they surface in wandb.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from aero_cfd.callbacks.aero_metrics import METRIC_PREFIX_LOSS, METRIC_SUFFIX_TARGET
from models.diffusion_abupt import DiffusionABUPT
from pydantic import Field, model_validator

from noether.core.schemas.lib import ConfiguredBy

from .dataspace_diffusion_chunked_eval import (
    DataspaceDiffusionChunkedEvalCallback,
    DataspaceDiffusionChunkedEvalCallbackConfig,
)


class DataspaceDiffusionUQCallbackConfig(DataspaceDiffusionChunkedEvalCallbackConfig):
    """Config for diffusion uncertainty-quantification eval.

    Inherits dataset_key, forward_properties, and the FM schedule from
    :class:`DataspaceDiffusionChunkedEvalCallbackConfig`. Requires a single
    ``sampling_steps`` entry (the multi-step sweep mode of the parent doesn't
    compose with per-geometry ensembling) and ``compute_forces=True`` so the
    batch carries the mesh tensors needed for Cd / Cl integration.

    Replaces the parent's anchor-chunked sampler with query-based inference
    (mirroring :class:`aero_cfd.callbacks.QueryInferenceCallback`): the first
    ``num_surface_anchors`` / ``num_volume_anchors`` positions are treated as
    training-size anchors (anchor self-attention sees only these) and the
    remainder are processed through the query branch in ``query_chunk_size``
    chunks at every FM Euler step.
    """

    name: Literal["DataspaceDiffusionUQCallback"] = "DataspaceDiffusionUQCallback"
    n_uq_samples: int = Field(5, ge=2)
    """Number of independent diffusion draws per geometry. Each draw runs the
    full FM Euler sampler; cost scales linearly."""
    scatter_max_points_per_sample: int = Field(2000, ge=1)
    """Upper bound on per-geometry points contributed to the pooled
    std-vs-|error| scatter. Capped to keep the rendered figure tractable."""
    num_surface_anchors: int = Field(..., ge=1)
    """Number of surface positions to treat as anchors per Euler step. Must
    match the anchor count the model was trained with."""
    num_volume_anchors: int = Field(..., ge=1)
    """Number of volume positions to treat as anchors per Euler step. Must
    match the anchor count the model was trained with."""
    query_chunk_size: int = Field(16384, ge=1)
    """Max query points per domain per forward pass within a single Euler step."""
    stl_root_path: Path | None = Field(None)
    """Root directory holding the raw DrivAerML STL files (e.g.
    ``/.../drivaerml/raw_surface_data``). When set, the spatial UQ renderer
    looks up ``<stl_root_path>/run_<design_id>/drivaer_<design_id>.stl`` for the
    pyvista overlay, with a fallback to ``<sample_uri>/drivaer_<design_id>.stl``
    if that's missing. **When unset, the spatial UQ plotting is skipped
    entirely** — std-vs-error scatter and Cd / Cl calibration still run."""

    @model_validator(mode="after")
    def _validate_uq(self) -> DataspaceDiffusionUQCallbackConfig:
        if len(self.sampling_steps) != 1:
            raise ValueError(
                "DataspaceDiffusionUQCallback requires exactly one sampling_steps entry "
                "(per-geometry ensembling is incompatible with the multi-step sweep mode)."
            )
        if not self.compute_forces:
            raise ValueError(
                "DataspaceDiffusionUQCallback requires compute_forces=True so the batch "
                "exposes surface_normals / surface_area / surface_position for Cd / Cl integration."
            )
        if self.chunked_inference:
            raise ValueError(
                "DataspaceDiffusionUQCallback uses query-based inference; set "
                "chunked_inference=False and use num_*_anchors / query_chunk_size instead."
            )
        return self


@ConfiguredBy(DataspaceDiffusionUQCallbackConfig)
class DataspaceDiffusionUQCallback(DataspaceDiffusionChunkedEvalCallback):
    """Diffusion UQ eval: per-geometry ensembling of FM samples via query-based inference."""

    def __init__(self, callback_config: DataspaceDiffusionUQCallbackConfig, **kwargs):
        super().__init__(callback_config, **kwargs)
        self._n_uq_samples: int = callback_config.n_uq_samples
        self._scatter_max_points: int = callback_config.scatter_max_points_per_sample
        self._num_anchors: dict[str, int] = {
            "surface": callback_config.num_surface_anchors,
            "volume": callback_config.num_volume_anchors,
        }
        self._query_chunk_size: int = callback_config.query_chunk_size
        self._stl_root_path: Path | None = callback_config.stl_root_path
        # Snapshot of the first geometry's per-point spatial UQ data (positions,
        # gt, mean, std, domain) per field, plus the dataset sample info needed
        # to resolve the STL path for the pyvista renderer. Populated on the
        # first sample seen by _compute_per_point_uq and rendered as a 4-panel
        # figure in process_results; cleared at the end of each eval pass.
        self._spatial_cache: dict[str, dict[str, Any]] = {}
        self._spatial_sample_info: dict[str, Any] | None = None

    def _run_model_inference(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Query-based FM Euler sampling — all ``n_uq_samples`` draws in one batched forward.

        Splits each domain's positions into the first ``num_*_anchors`` (treated
        as training-size anchors) and the remainder (queries). Inputs are then
        repeated ``n_uq_samples`` times along the batch dim so a single
        ``schedule.sample_joint`` call produces all draws at once (each batch
        slot gets its own random initial state from ``torch.randn``). Flat-concat
        geometry is replicated along the flat dim with offset supernode indices
        and an N_uq-spanning ``geometry_batch_idx``.

        Each Euler step runs one forward per query chunk, with the noisy
        anchor features and anchor self-attention state shared across chunks.
        Anchor velocity is taken from the first chunk — anchor outputs vary
        slightly across chunks due to decoder self-attention over
        ``[anchors, queries]``, and :class:`aero_cfd.callbacks.QueryInferenceCallback`
        uses the same convention. Geometry encoding is cached once across all
        steps + chunks.

        Returns per-field tensors with batch dim ``n_uq_samples`` (one draw per
        slot); :meth:`process_data` splits them back into a per-draw list.
        """
        model = self.model
        assert isinstance(model, DiffusionABUPT), (
            f"DataspaceDiffusionUQCallback expects DiffusionABUPT, got {type(model).__name__}"
        )
        device = next(model.parameters()).device
        schedule = self._schedule.to(device)

        domain_names: list[str] = list(model.domain_names)
        n_uq = self._n_uq_samples

        anchor_positions: dict[str, torch.Tensor] = {}
        query_positions: dict[str, torch.Tensor] = {}
        for name in domain_names:
            all_pos = batch[f"{name}_anchor_position"]
            n_a = self._num_anchors[name]
            if all_pos.shape[1] < n_a:
                raise ValueError(
                    f"domain {name!r}: batch has {all_pos.shape[1]} positions but num_{name}_anchors={n_a}"
                )
            # Repeat along the batch dim so the n_uq draws are independent forwards in a single batch.
            anchor_positions[name] = all_pos[:, :n_a].repeat(n_uq, 1, 1)
            query_positions[name] = all_pos[:, n_a:].repeat(n_uq, 1, 1)

        field_slices = {name: model.data_specs.domains[name].output_dims.field_slices for name in domain_names}
        domain_shapes = [
            (
                n_uq,
                anchor_positions[name].shape[1] + query_positions[name].shape[1],
                model.data_specs.domains[name].output_dims.total_dim,
            )
            for name in domain_names
        ]

        # Flat-concat geometry: replicate along the flat dim n_uq times with offset
        # supernode indices so each batch slot owns its own geometry copy. The encoder
        # then produces a (n_uq, n_super, D) encoding that drives all draws in parallel.
        geo_pos = batch["geometry_position"]
        geo_super_idx = batch["geometry_supernode_idx"]
        n_geom = geo_pos.shape[0]
        n_super = geo_super_idx.shape[0]
        arange_uq = torch.arange(n_uq, device=device)
        geo_inputs = {
            "geometry_position": geo_pos.repeat(n_uq, 1),
            "geometry_supernode_idx": geo_super_idx.repeat(n_uq) + arange_uq.repeat_interleave(n_super) * n_geom,
            "geometry_batch_idx": arange_uq.repeat_interleave(n_geom),
        }
        geometry_cache: dict[str, torch.Tensor] = {}
        cs = self._query_chunk_size

        def joint_fn(xt_list, t, _cond):
            anchor_feat: dict[str, torch.Tensor] = {}
            query_feat: dict[str, torch.Tensor] = {}
            for name, xt in zip(domain_names, xt_list, strict=True):
                n_a = anchor_positions[name].shape[1]
                anchor_feat[name] = xt[:, :n_a]
                query_feat[name] = xt[:, n_a:]

            chunk_counts = [
                math.ceil(query_feat[name].shape[1] / cs) for name in domain_names if query_feat[name].shape[1] > 0
            ]
            n_chunks = max(chunk_counts) if chunk_counts else 1

            anchor_v: dict[str, torch.Tensor] | None = None
            query_v_chunks: dict[str, list[torch.Tensor]] = {name: [] for name in domain_names}

            for chunk_idx in range(n_chunks):
                chunk_q_pos: dict[str, torch.Tensor] = {}
                chunk_q_feat: dict[str, torch.Tensor] = {}
                for name in domain_names:
                    n_q = query_feat[name].shape[1]
                    if n_q == 0:
                        continue
                    q_start = chunk_idx * cs
                    if q_start >= n_q:
                        continue
                    q_end = min(q_start + cs, n_q)
                    chunk_q_pos[name] = query_positions[name][:, q_start:q_end]
                    chunk_q_feat[name] = query_feat[name][:, q_start:q_end]

                backbone_kwargs: dict[str, Any] = dict(
                    domain_anchor_positions=anchor_positions,
                    domain_anchor_features=anchor_feat,
                    conditioning_inputs={"timestep": t.view(-1, 1)},
                    kv_cache=geometry_cache,
                )
                if chunk_q_pos:
                    backbone_kwargs["domain_query_positions"] = chunk_q_pos
                    backbone_kwargs["domain_query_features"] = chunk_q_feat
                if "geometry_encoding" not in geometry_cache:
                    backbone_kwargs.update(geo_inputs)

                predictions, new_cache = model.backbone(**backbone_kwargs)

                if "geometry_encoding" not in geometry_cache:
                    geometry_cache["geometry_encoding"] = new_cache["geometry_encoding"]
                    geometry_cache["geometry_rope"] = new_cache["geometry_rope"]

                if anchor_v is None:
                    anchor_v = {
                        name: torch.cat(
                            [predictions[f"{name}_{field}"] for field in field_slices[name]],
                            dim=-1,
                        )
                        for name in domain_names
                    }
                for name in chunk_q_pos:
                    query_v_chunks[name].append(
                        torch.cat(
                            [predictions[f"query_{name}_{field}"] for field in field_slices[name]],
                            dim=-1,
                        )
                    )

            assert anchor_v is not None
            return [
                torch.cat([anchor_v[name], *query_v_chunks[name]], dim=1) if query_v_chunks[name] else anchor_v[name]
                for name in domain_names
            ]

        with self.trainer.autocast_context, torch.no_grad():
            x_list = schedule.sample_joint(
                domain_shapes,
                joint_fn,
                steps=self._current_sampling_steps,
            )

        result: dict[str, torch.Tensor] = {}
        for name, x in zip(domain_names, x_list, strict=True):
            for field, sl in field_slices[name].items():
                result[f"{name}_{field}"] = x[..., sl]
        return result

    def process_data(self, batch: dict[str, torch.Tensor], **_) -> dict[str, torch.Tensor]:
        """Run all ``n_uq_samples`` FM draws in a single batched forward and reduce to UQ stats.

        :meth:`_run_model_inference` returns per-field tensors with batch dim
        ``n_uq_samples``; we split them back into a per-draw list so the
        downstream reductions keep their original per-draw interface.

        Returns per-sample tensors shaped so that
        :meth:`PeriodicDataIteratorCallback._collate_result` stacks them into
        per-geometry rows in ``process_results``:

        * 0-d scalars (gt_cd, gt_cl, ``<field>_uq_corr``) stack to ``(S,)``.
        * ``(1, N)`` Cd / Cl draw tensors concat to ``(S, N)``.
        * ``(1, K, 2)`` per-field scatter pairs concat to ``(S, K, 2)``.
        """
        out = self._run_model_inference(batch)
        per_draw: list[dict[str, torch.Tensor]] = [
            {k: v[i : i + 1] for k, v in out.items()} for i in range(self._n_uq_samples)
        ]

        result: dict[str, torch.Tensor] = {}
        result.update(self._compute_per_point_uq(batch, per_draw))
        result.update(self._compute_force_uq(batch, per_draw))
        return result

    def _compute_per_point_uq(
        self,
        batch: dict[str, torch.Tensor],
        per_draw: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Per-field std vs |error|: Pearson correlation, scatter subsample, mean-draw L2 error."""
        dataset = self.data_container.get_dataset(self.dataset_key)
        out: dict[str, torch.Tensor] = {}

        for field in self.evaluation_modes:
            target_key = f"{field}{METRIC_SUFFIX_TARGET}"
            if field not in per_draw[0] or target_key not in batch:
                continue

            # (N, B, n_pts, n_dims) — denormalize each draw separately to apply
            # the inverse transform correctly (mean/std of normalized space != normalized mean/std).
            samples = torch.stack(
                [dataset.denormalize(field, d[field]) for d in per_draw],
                dim=0,
            )
            target = dataset.denormalize(field, batch[target_key])

            # Relative L2 error of the ensemble mean (per-channel mean over draws),
            # matching the L2ERR metric the parent AeroMetricsCallback emits for
            # single-draw predictions so the two are directly comparable.
            mean_pred = samples.mean(dim=0)
            target_norm = target.norm()
            if float(target_norm) > 1e-8:
                out[f"{field}_uq_mean_l2err"] = (mean_pred - target).norm() / target_norm
            else:
                out[f"{field}_uq_mean_l2err"] = torch.tensor(float("nan"))

            # Reduce vector fields to magnitude so std and |error| are per-point scalars
            # — same projection the notebook uses for the WSS panel.
            if samples.shape[-1] > 1:
                samples_scalar = samples.norm(dim=-1)
                target_scalar = target.norm(dim=-1)
            else:
                samples_scalar = samples.squeeze(-1)
                target_scalar = target.squeeze(-1)

            mean = samples_scalar.mean(dim=0)
            std = samples_scalar.std(dim=0, unbiased=False)
            abs_err = (mean - target_scalar).abs()

            # Cache the first geometry's spatial snapshot per surface field
            # for the 4-panel gt/mean/std/|err| STL render. Positions follow
            # the same [anchors, queries] ordering as the predicted fields
            # (see _run_model_inference final concat). Volume fields are
            # skipped — the pyvista overlay only makes sense on the surface
            # mesh. Also skipped when no STL root is configured.
            if self._stl_root_path is not None and field.startswith("surface_") and field not in self._spatial_cache:
                domain = "surface"
                pos_key = f"{domain}_anchor_position"
                if pos_key in batch:
                    self._spatial_cache[field] = {
                        "domain": domain,
                        "positions": batch[pos_key][0].detach().cpu().float().numpy(),
                        "target": target_scalar[0].detach().cpu().float().numpy(),
                        "mean": mean[0].detach().cpu().float().numpy(),
                        "std": std[0].detach().cpu().float().numpy(),
                    }
                    # Cache the dataset sample info once so the STL renderer
                    # can resolve the raw mesh path. Mirrors the lookup in
                    # _compute_force_uq.
                    if self._spatial_sample_info is None and "index" in batch:
                        try:
                            dataset = self.data_container.get_dataset(self.dataset_key)
                            sample_idx = int(batch["index"].squeeze().item())
                            info = dataset.sample_info(sample_idx)
                            self._spatial_sample_info = {
                                "run_dir": Path(info["sample_uri"]),
                                "design_id": info["design_id"],
                            }
                        except Exception as exc:
                            self.logger.debug(f"Spatial UQ: could not resolve sample_info: {exc}")

            std_flat = std.flatten().detach().cpu().float()
            err_flat = abs_err.flatten().detach().cpu().float()

            out[f"{field}_uq_corr"] = _pearson(std_flat, err_flat)

            n_sub = min(self._scatter_max_points, std_flat.numel())
            idx = torch.randperm(std_flat.numel())[:n_sub]
            pairs = torch.stack([std_flat[idx], err_flat[idx]], dim=-1).unsqueeze(0)
            out[f"{field}_uq_pairs"] = pairs

        return out

    def _compute_force_uq(
        self,
        batch: dict[str, torch.Tensor],
        per_draw: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Per-draw Cd / Cl + GT Cd / Cl for this geometry.

        Loads full-resolution GT pressure / shear from disk to match the
        ``compute_forces`` path in the parent (batch targets are subsampled by
        the pipeline). Each predicted draw is matched to the mesh via a single
        shared KDTree query — geometry is identical across draws so we only
        build the tree once.
        """
        surface_normals = batch.get("surface_normals")
        surface_areas = batch.get("surface_area")
        mesh_positions = batch.get("surface_position")
        pred_positions = batch.get("surface_anchor_position")

        if surface_normals is None or surface_areas is None or mesh_positions is None or pred_positions is None:
            self.logger.warning(
                "Skipping Cd / Cl UQ: surface_normals / surface_area / surface_position "
                "missing from batch. Ensure these fields are not excluded in the dataset config."
            )
            return {}

        surface_normals = surface_normals.cpu().squeeze(0).float()
        surface_areas = surface_areas.cpu().squeeze(0).float()
        mesh_positions = mesh_positions.cpu().squeeze(0).float()
        pred_positions_cpu = pred_positions.cpu().squeeze(0)

        dataset = self.data_container.get_dataset(self.dataset_key)
        sample_idx = batch["index"].squeeze().item()
        info = dataset.sample_info(sample_idx)
        run_dir = Path(info["sample_uri"])
        design_id = info["design_id"]

        ref_csv = run_dir / f"geo_ref_{design_id}.csv"
        if ref_csv.exists():
            import pandas as pd

            ref_area = float(pd.read_csv(ref_csv)["aRef"][0])
            flow = self._FlowConditions(reference_area=ref_area)
        else:
            flow = self._FlowConditions()

        gt_pressure_path = run_dir / "surface_pressure.pt"
        gt_shear_path = run_dir / "surface_wallshearstress.pt"
        if not gt_pressure_path.exists() or not gt_shear_path.exists():
            self.logger.debug(f"Skipping Cd / Cl UQ for sample {sample_idx}: missing GT files")
            return {}

        gt_pressure = torch.load(gt_pressure_path, map_location="cpu", weights_only=True).float()
        gt_shear = torch.load(gt_shear_path, map_location="cpu", weights_only=True).float()
        if gt_pressure.ndim == 2 and gt_pressure.shape[-1] == 1:
            gt_pressure = gt_pressure.squeeze(-1)

        tree = self._cKDTree(mesh_positions.numpy())
        _, matched = tree.query(pred_positions_cpu.numpy())
        matched_normals = surface_normals[matched]
        matched_areas = surface_areas[matched]

        gt = self._compute_force_coefficients(
            gt_pressure[matched], gt_shear[matched], matched_normals, matched_areas, flow
        )

        cd_list: list[float] = []
        cl_list: list[float] = []
        for draw in per_draw:
            pred_p = self.dataset_normalizers["surface_pressure"].inverse(draw["surface_pressure"].cpu()).squeeze(0)
            pred_f = self.dataset_normalizers["surface_friction"].inverse(draw["surface_friction"].cpu()).squeeze(0)
            if pred_p.ndim == 2 and pred_p.shape[-1] == 1:
                pred_p = pred_p.squeeze(-1)
            coeffs = self._compute_force_coefficients(pred_p, pred_f, matched_normals, matched_areas, flow)
            cd_list.append(coeffs.cd)
            cl_list.append(coeffs.cl)

        return {
            "cd_gt": torch.tensor(gt.cd, dtype=torch.float32),
            "cl_gt": torch.tensor(gt.cl, dtype=torch.float32),
            "cd_draws": torch.tensor(cd_list, dtype=torch.float32).unsqueeze(0),
            "cl_draws": torch.tensor(cl_list, dtype=torch.float32).unsqueeze(0),
        }

    def process_results(self, results: Any, **_) -> None:
        """Aggregate per-geometry ensembles into calibration metrics + figures."""
        if not isinstance(results, dict) or not results:
            self.logger.warning(f"No UQ results for dataset '{self.dataset_key}'")
            return

        self._log_per_field_scalars(results)
        self._log_force_calibration(results)
        self._log_std_vs_err_scatter(results)
        self._log_field_spatial_uq()
        self._spatial_cache.clear()
        self._spatial_sample_info = None

    _PER_FIELD_SCALAR_SUFFIXES = ("_uq_corr", "_uq_mean_l2err")

    def _log_per_field_scalars(self, results: dict[str, torch.Tensor]) -> None:
        for key, value in results.items():
            if not key.endswith(self._PER_FIELD_SCALAR_SUFFIXES):
                continue
            metric_key = f"{METRIC_PREFIX_LOSS}{self.dataset_key}/{key}"
            self.writer.add_scalar(
                key=metric_key,
                value=value.mean(),
                logger=self.logger,
                format_str=".4f",
            )

    def _log_force_calibration(self, results: dict[str, torch.Tensor]) -> None:
        for force in ("cd", "cl"):
            gt_key = f"{force}_gt"
            draws_key = f"{force}_draws"
            if gt_key not in results or draws_key not in results:
                continue

            gt = results[gt_key].float()
            draws = results[draws_key].float()
            mu = draws.mean(dim=-1)
            sd = draws.std(dim=-1, unbiased=False)
            err = (mu - gt).abs()
            lo = torch.quantile(draws, 0.025, dim=-1)
            hi = torch.quantile(draws, 0.975, dim=-1)

            r2 = _r2_score(gt, mu)
            coverage_1sig = (err <= sd).float().mean().item()
            coverage_ci95 = ((gt >= lo) & (gt <= hi)).float().mean().item()
            mean_z_abs = (err / (sd + 1e-12)).mean().item()
            mean_std = sd.mean().item()

            base = f"{METRIC_PREFIX_LOSS}{self.dataset_key}/{force}_uq"
            self._add_scalar(f"{base}_r2", r2)
            self._add_scalar(f"{base}_coverage_1sig", coverage_1sig)
            self._add_scalar(f"{base}_coverage_ci95", coverage_ci95)
            self._add_scalar(f"{base}_mean_z_abs", mean_z_abs)
            self._add_scalar(f"{base}_mean_std", mean_std)

            fig = _plot_force_calibration(
                gt.cpu().numpy(),
                mu.cpu().numpy(),
                sd.cpu().numpy(),
                lo.cpu().numpy(),
                hi.cpu().numpy(),
                force.upper(),
                r2,
                coverage_1sig,
                coverage_ci95,
            )
            self._publish_figure(fig, f"{force}_calibration")

    def _log_std_vs_err_scatter(self, results: dict[str, torch.Tensor]) -> None:
        for field in self.evaluation_modes:
            pairs_key = f"{field}_uq_pairs"
            if pairs_key not in results:
                continue
            pairs = results[pairs_key]
            if pairs.ndim == 3:
                pairs = pairs.reshape(-1, 2)
            std = pairs[:, 0].cpu().numpy()
            err = pairs[:, 1].cpu().numpy()
            pooled_corr = _pearson(torch.from_numpy(std), torch.from_numpy(err)).item()
            fig = _plot_std_vs_err(std, err, field, pooled_corr)
            self._publish_figure(fig, f"{field}_uq_scatter")

    def _log_field_spatial_uq(self) -> None:
        """Render a 4-panel surface view (gt / mean / std / |error|) per
        surface field from the first geometry's cached snapshot.

        Surface fields render on the raw DrivAerML STL via pyvista (k-NN
        interpolated onto the mesh vertices). Volume fields are not plotted —
        the STL overlay only makes sense on the surface mesh. Skipped
        entirely when ``stl_root_path`` isn't configured or no STL was
        resolvable for the cached geometry; falls back to a matplotlib 3D
        scatter only if the pyvista render itself fails.
        """
        if self._stl_root_path is None or not self._spatial_cache:
            return
        stl_path = self._resolve_stl_path()
        if stl_path is None:
            return

        for field, data in self._spatial_cache.items():
            try:
                fig = _plot_field_uq_stl(
                    positions=data["positions"],
                    target=data["target"],
                    mean=data["mean"],
                    std=data["std"],
                    stl_path=stl_path,
                    field=field,
                )
            except Exception as exc:
                self.logger.warning(f"Spatial UQ (STL) for '{field}' failed: {exc}; falling back to scatter")
                fig = _plot_field_uq_spatial(
                    positions=data["positions"],
                    target=data["target"],
                    mean=data["mean"],
                    std=data["std"],
                    field=field,
                )
            self._publish_figure(fig, f"{field}_uq_spatial")

    def _resolve_stl_path(self) -> Path | None:
        """Build the path to the raw DrivAerML STL for the cached geometry.

        Tries ``stl_root_path / run_<id> / drivaer_<id>.stl`` first if the
        config field is set, then falls back to
        ``sample_uri / drivaer_<id>.stl`` for setups where STLs live next to
        the rest of the per-run files.
        """
        if self._spatial_sample_info is None:
            return None
        run_dir: Path = self._spatial_sample_info["run_dir"]
        design_id = self._spatial_sample_info["design_id"]

        candidates: list[Path] = []
        if self._stl_root_path is not None:
            candidates.append(self._stl_root_path / f"run_{design_id}" / f"drivaer_{design_id}.stl")
        candidates.append(run_dir / f"drivaer_{design_id}.stl")

        for candidate in candidates:
            if candidate.exists():
                return candidate
        self.logger.debug(f"Spatial UQ: STL not found at any of {candidates}")
        return None

    def _add_scalar(self, key: str, value: float) -> None:
        self.writer.add_scalar(
            key=key,
            value=torch.tensor(value),
            logger=self.logger,
            format_str=".4f",
        )

    def _publish_figure(self, fig, name: str) -> None:
        cp = self.trainer.update_counter.cur_iteration
        out_dir = self.checkpoint_writer.path_provider.run_output_path / "uq" / self.dataset_key
        out_dir.mkdir(parents=True, exist_ok=True)
        uri = out_dir / f"{name}_cp={cp}.png"
        fig.savefig(uri, dpi=150, bbox_inches="tight")
        self.writer.add_nonscalar(
            key=f"uq/{self.dataset_key}/{name}",
            value=fig,
        )

        import matplotlib.pyplot as plt

        plt.close(fig)


def _pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.flatten().float()
    b = b.flatten().float()
    am = a - a.mean()
    bm = b - b.mean()
    denom = am.norm() * bm.norm()
    if float(denom) <= 1e-12:
        return torch.tensor(float("nan"))
    return (am * bm).sum() / denom


def _r2_score(target: torch.Tensor, pred: torch.Tensor) -> float:
    ss_res = float(((target - pred) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _plot_force_calibration(gt, mu, sd, lo, hi, label, r2, cov_1sig, cov_ci95):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    # Asymmetric error bars from the empirical 95% CI around the per-geometry mean.
    yerr_low = mu - lo
    yerr_high = hi - mu
    ax.errorbar(
        gt,
        mu,
        yerr=[yerr_low, yerr_high],
        fmt="o",
        ms=5,
        alpha=0.7,
        capsize=3,
        color="tab:blue",
        label="mean +/- 95% CI",
    )
    lo_lim = float(min(gt.min(), lo.min()))
    hi_lim = float(max(gt.max(), hi.max()))
    ax.plot([lo_lim, hi_lim], [lo_lim, hi_lim], "k--", alpha=0.5, label="y=x")
    ax.set_xlabel(f"target {label}")
    ax.set_ylabel(f"pred {label} (mean +/- CI)")
    ax.set_title(f"{label} calibration\nR^2={r2:.3f}  cov@1sigma={cov_1sig:.2f}  cov@95% CI={cov_ci95:.2f}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_std_vs_err(std, err, field, corr):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(std, err, s=2, alpha=0.3, color="tab:blue")
    ax.set_xlabel("predictive std (across draws)")
    ax.set_ylabel("|mean - target|")
    ax.set_title(f"{field}: std vs |error|  (Pearson r={corr:.3f})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_field_uq_stl(
    positions: np.ndarray,
    target: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    stl_path: Path,
    field: str,
    view: str = "front",
    angle: float = 30.0,
    elevation: float = 10.0,
    window_size: tuple[int, int] = (600, 500),
    interp_k: int = 16,
) -> Any:
    """4-panel pyvista render of the car STL — gt, pred mean, ensemble std,
    |pred - gt|.

    Mirrors ``viz.plot_field_uq_stl``: bbox-aligns the anchor cloud to the STL
    bounds, k-NN-interpolates anchor scalars onto STL vertices with a
    Gaussian-weighted kernel, then composites four off-screen pyvista
    screenshots into a matplotlib figure (so the existing ``_publish_figure``
    path handles disk + wandb logging).
    """
    import math

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import pyvista as pv
    from scipy.spatial import cKDTree

    mesh = pv.read(str(stl_path))

    # Bbox alignment: anchor positions come out of the pipeline position-
    # normalized; the STL is raw mm. Match bboxes (biased when the sample
    # cloud underfills the true mesh extent, but exact alignment would need
    # the dataset's PositionNormalizer which we don't have here).
    a_min = positions.min(axis=0)
    a_max = positions.max(axis=0)
    s_min = mesh.points.min(axis=0)
    s_max = mesh.points.max(axis=0)
    denom = np.where(s_max - s_min > 1e-9, s_max - s_min, 1.0)
    scale = (a_max - a_min) / denom
    shift = a_min - s_min * scale
    query_pts = mesh.points * scale + shift

    kdt = cKDTree(positions)
    dists, idxs = kdt.query(query_pts, k=max(interp_k, 1))
    if interp_k == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    # Gaussian kernel with per-query bandwidth = mean of k distances.
    sigma = dists.mean(axis=1, keepdims=True) + 1e-12
    weights = np.exp(-0.5 * (dists / sigma) ** 2)
    weights /= weights.sum(axis=1, keepdims=True)

    def _interp(arr: np.ndarray) -> np.ndarray:
        return (arr[idxs] * weights).sum(axis=1)

    error = np.abs(target - mean)
    vmin, vmax = float(target.min()), float(target.max())
    # Robust upper clip: 98th percentile so a handful of outlier points don't
    # crush the rest of the field into the darkest band.
    umax = float(max(np.percentile(std, 98), 1e-12))
    emax = float(max(np.percentile(error, 98), 1e-12))

    eps = 1e-12
    corr = float(np.corrcoef(std.ravel(), error.ravel())[0, 1]) if std.std() > eps else float("nan")

    mesh["gt"] = _interp(target)
    mesh["pred"] = _interp(mean)
    mesh["std"] = _interp(std)
    mesh["err"] = _interp(error)

    panels = [
        ("gt", f"gt {field}", "RdBu_r", (vmin, vmax)),
        ("pred", f"pred mean {field}", "RdBu_r", (vmin, vmax)),
        ("std", f"uq (std), r={corr:.3f}", "YlOrRd", (0.0, umax)),
        ("err", "|pred - gt|", "Reds", (0.0, emax)),
    ]

    # Camera. DrivAerML convention: car faces -X (nose at xmin), so "front"
    # means camera on the -X side looking toward +X. ``angle`` rotates around
    # Z by that many degrees off head-on; ``elevation`` tilts up above XY.
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    span = max(xmax - xmin, ymax - ymin, zmax - zmin) * 1.3
    th = math.radians(angle)
    el = math.radians(elevation)
    if view == "front":
        base_dir = (-math.cos(th) * math.cos(el), math.sin(th) * math.cos(el), math.sin(el))
    elif view == "rear":
        base_dir = (math.cos(th) * math.cos(el), math.sin(th) * math.cos(el), math.sin(el))
    elif view == "side":
        base_dir = (math.sin(th) * math.cos(el), math.cos(th) * math.cos(el), math.sin(el))
    elif view == "top":
        base_dir = (0.0, 0.0, 1.0)
    else:
        base_dir = (-0.7, 0.7, 0.3)
    cam_pos = (cx + base_dir[0] * span, cy + base_dir[1] * span, cz + base_dir[2] * span)
    up = (0, 0, 1) if view != "top" else (1, 0, 0)
    camera = [cam_pos, (cx, cy, cz), up]

    images = []
    for scalar, _title, cmap, clim in panels:
        p = pv.Plotter(off_screen=True, window_size=window_size)
        p.add_mesh(mesh, scalars=scalar, cmap=cmap, clim=clim, show_scalar_bar=False)
        p.set_background("white")
        p.camera_position = camera
        images.append(p.screenshot(return_img=True))
        p.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, img, (_, title, cmap, clim) in zip(axes.ravel(), images, panels, strict=True):
        ax.imshow(img)
        ax.set_title(title)
        ax.set_axis_off()
        norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6)

    fig.suptitle(field, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _plot_field_uq_spatial(
    positions: np.ndarray,
    target: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    field: str,
    max_points: int = 20_000,
) -> Any:
    """4-panel 3D scatter — gt, predicted mean, ensemble std, |pred - gt|.

    Fallback renderer used for volume fields (no STL available) and when the
    pyvista STL path fails. Mirrors ``viz.plot_field_uq``; color bars are
    shared between gt + mean (symmetric data range) and free-floating on
    std + |error| (sequential colormaps from 0).
    """
    import matplotlib.pyplot as plt

    if positions.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(positions.shape[0], size=max_points, replace=False)
        positions = positions[idx]
        target = target[idx]
        mean = mean[idx]
        std = std[idx]

    error = np.abs(target - mean)
    vmin, vmax = float(target.min()), float(target.max())
    eps = 1e-12
    smax = float(max(std.max(), eps))
    emax = float(max(error.max(), eps))
    corr = float(np.corrcoef(std.ravel(), error.ravel())[0, 1]) if std.std() > eps else float("nan")

    fig, axes = plt.subplots(2, 2, figsize=(13, 12), subplot_kw={"projection": "3d"})
    flat_axes = axes.ravel()
    panels = [
        (flat_axes[0], target, f"gt {field}", "RdBu_r", (vmin, vmax)),
        (flat_axes[1], mean, f"pred mean {field}", "RdBu_r", (vmin, vmax)),
        (flat_axes[2], std, f"uq (std), r={corr:.3f}", "viridis", (0.0, smax)),
        (flat_axes[3], error, "|pred - gt|", "magma", (0.0, emax)),
    ]
    for ax, vals, title, cmap, vm in panels:
        sc = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=vals,
            cmap=cmap,
            s=0.5,
            alpha=0.85,
            vmin=vm[0],
            vmax=vm[1],
        )
        ax.set_title(title)
        fig.colorbar(sc, ax=ax, shrink=0.55)
        ax.set_axis_off()

    fig.suptitle(field, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig
