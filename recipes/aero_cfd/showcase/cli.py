#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

import torch
import typer
from rich.console import Console

from showcase.model_configs import (
    ABUPT_MODEL_KIND,
    FIELD_WEIGHTS,
    MODEL_SIZES,
    TRAINER_KIND,
    ModelSize,
)


class Tracker(str, Enum):
    """Supported experiment trackers."""

    wandb = "wandb"
    tensorboard = "tensorboard"
    trackio = "trackio"


TRACKER_KINDS: dict[str, str] = {
    "wandb": "noether.core.trackers.WandBTracker",
    "tensorboard": "noether.core.trackers.TensorboardTracker",
    "trackio": "noether.core.trackers.TrackioTracker",
}

# Maps domain prefix -> position key saved from the batch.
DOMAIN_POSITION_KEYS: dict[str, str] = {
    "surface": "surface_anchor_position",
    "volume": "volume_anchor_position",
}

app = typer.Typer(
    name="abupt-showcase",
    pretty_exceptions_short=True,
    pretty_exceptions_show_locals=False,
    help="""AB-UPT + DrivAerML Showcase.

    Single entry point for training, evaluating, and exporting AB-UPT models on the DrivAerML aerodynamic CFD dataset.

    Must be run from the ``recipes/aero_cfd/`` directory::

        # Train:
        python -m showcase.cli train --dataset-root /data/drivaerml --output-path /outputs

        # Evaluate (anchor-resolution metrics + save predictions):
        python -m showcase.cli evaluate --dataset-root /data/drivaerml \\
            --output-path /outputs --run-id 2026-04-09_abc12

        # Dense query inference + VTK point clouds for ParaView or a custom VTK viewer:
        python -m showcase.cli evaluate --dataset-root /data/drivaerml \\
            --output-path /outputs --run-id 2026-04-09_abc12 \\
            --query-inference --num-inference-surface-points 20000 --export-vtk

        # Export a single sample as VTK:
        python -m showcase.cli export-vtk \\
            --predictions-path /outputs/2026-04-09_abc12/eval/predictions/sample_0000.pt \\
            --output-path out.vtp
    """,
    no_args_is_help=True,
)
console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_device(accelerator: str) -> str:
    from noether.core.distributed.utils import accelerator_to_device

    return accelerator_to_device(accelerator)


def _get_preset():  # noqa: ANN202
    from aero_cfd.presets import DrivAerMLPreset

    return DrivAerMLPreset()


def _build_eval_config(
    *,
    preset,
    size_config,
    dataset_root: str,
    output_path: str,
    split: str,
    accelerator: str,
    run_id: str,
    checkpoint: str,
    callbacks: list,
    extra_datasets: dict | None = None,
    precision: str = "float32",
):
    """Build a ConfigSchema for evaluation (shared by standard and query modes)."""
    return preset.build_config(
        model_kind=ABUPT_MODEL_KIND,
        model_params=size_config.model_params,
        trainer_kind=TRAINER_KIND,
        trainer_params=dict(field_weights=FIELD_WEIGHTS, precision=precision),
        dataset_root=dataset_root,
        output_path=output_path,
        datasets={split: split},
        extra_datasets=extra_datasets or {},
        max_epochs=1,
        accelerator=accelerator,
        callbacks_override=callbacks,
        resume_run_id=run_id,
        resume_checkpoint=checkpoint,
    )


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@app.command()
def train(
    dataset_root: Annotated[str, typer.Option(help="Path to the DrivAerML dataset.")],
    output_path: Annotated[str, typer.Option(help="Path to store training outputs.")],
    model_size: Annotated[ModelSize, typer.Option(help="Model size tier.")] = ModelSize.small,
    accelerator: Annotated[str, typer.Option(help="Accelerator: cpu, gpu, or mps.")] = "cpu",
    max_epochs: Annotated[int, typer.Option(help="Maximum training epochs.")] = 100,
    eval_every_n_epochs: Annotated[int, typer.Option(help="Compute eval metrics every N epochs.")] = 10,
    tracker: Annotated[Tracker | None, typer.Option(help="Experiment tracker: wandb, tensorboard, or trackio.")] = None,
    tracker_project: Annotated[str | None, typer.Option(help="Project name for the tracker.")] = None,
    precision: Annotated[str, typer.Option(help="Training precision: float32, float16, or bfloat16.")] = "float32",
    resume_run_id: Annotated[str | None, typer.Option(help="Run ID to resume training from.")] = None,
    resume_checkpoint: Annotated[str, typer.Option(help="Checkpoint tag to resume from.")] = "latest",
    compute_forces: bool = typer.Option(
        False,
        "--compute-forces",
        help="Compute Cd/Cl errors during eval callbacks (logged to tracker).",
    ),
) -> None:
    from aero_cfd.callbacks import AeroMetricsCallbackConfig

    from noether.core.schemas.callbacks import CheckpointCallbackConfig
    from noether.training.runners import HydraRunner

    size_config = MODEL_SIZES[model_size.value]
    preset = _get_preset()
    preset.pipeline_model_overrides[ABUPT_MODEL_KIND].update(size_config.pipeline_overrides)

    eval_callback = AeroMetricsCallbackConfig(
        dataset_key="val",
        every_n_epochs=eval_every_n_epochs,
        forward_properties=preset.forward_properties(ABUPT_MODEL_KIND),
        compute_forces=compute_forces,
    )

    tracker_config = None
    if tracker is not None:
        tracker_config = {"kind": TRACKER_KINDS[tracker.value]}
        if tracker_project:
            tracker_config["project"] = tracker_project

    resume_kwargs = {}
    if resume_run_id is not None:
        # ResumeInitializer requires both model and optimizer checkpoints.
        # Runs created before optimizer saving was enabled won't have *_optim.th files.
        checkpoint_dir = Path(output_path) / resume_run_id / "checkpoints"
        optim_file = checkpoint_dir / f"ab_upt_cp={resume_checkpoint}_optim.th"
        if not optim_file.exists():
            available = sorted(f.name for f in checkpoint_dir.glob("*_optim.th")) if checkpoint_dir.exists() else []
            console.print(f"[red]Cannot resume: optimizer checkpoint not found at {optim_file}[/red]")
            if available:
                console.print(f"[yellow]Available optimizer checkpoints: {available}[/yellow]")
                console.print("[yellow]Try a different --resume-checkpoint tag.[/yellow]")
            else:
                console.print(
                    "[yellow]This run has no optimizer checkpoints. "
                    "Only runs started with the current CLI save optimizer state.[/yellow]"
                )
                console.print(
                    "[yellow]Start a fresh training run, or use 'evaluate' to load model weights only.[/yellow]"
                )
            raise typer.Exit(1)
        resume_kwargs["resume_run_id"] = resume_run_id
        resume_kwargs["resume_checkpoint"] = resume_checkpoint

    # Build standard callbacks but with optimizer saving enabled for resumability.
    standard_callbacks = preset.standard_callbacks()
    for callback in standard_callbacks:
        if isinstance(callback, CheckpointCallbackConfig):
            callback.save_optim = True

    config = preset.build_config(
        model_kind=ABUPT_MODEL_KIND,
        model_params=size_config.model_params,
        trainer_kind=TRAINER_KIND,
        trainer_params=dict(field_weights=FIELD_WEIGHTS, precision=precision),
        dataset_root=dataset_root,
        output_path=output_path,
        datasets=["train", "val", "test"],
        max_epochs=max_epochs,
        accelerator=accelerator,
        callbacks_override=standard_callbacks + [eval_callback],
        tracker=tracker_config,
        **resume_kwargs,
    )

    if compute_forces:
        # Only include force fields for the val split (eval callback), not train.
        for key in ("val", "test"):
            ds_config = config.datasets.get(key)
            if ds_config is not None and ds_config.excluded_properties:
                ds_config.excluded_properties -= {"surface_normals", "surface_area"}

    console.print(f"[bold]Training AB-UPT ({model_size.value}) on DrivAerML[/bold]")
    console.print(
        f"\tHidden dim: {size_config.model_params['hidden_dim']}, Heads: {size_config.model_params['num_heads']}"
    )
    console.print(f"\tDecoder blocks/domain: {size_config.model_params['num_domain_decoder_blocks']}")
    console.print(f"\tEpochs: {max_epochs}")
    console.print(f"\tOutput: {output_path}")

    HydraRunner().main(device=_get_device(accelerator), config=config)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


@app.command()
def evaluate(
    dataset_root: Annotated[str, typer.Option(help="Path to the DrivAerML dataset.")],
    output_path: Annotated[str, typer.Option(help="Root output directory (same as used for training).")],
    run_id: Annotated[str, typer.Option(help="Run ID of the training run to evaluate.")],
    checkpoint: Annotated[
        str, typer.Option(help="Checkpoint tag: 'latest', 'best_model.loss.test.total', or 'E10', etc.")
    ] = "latest",
    model_size: Annotated[ModelSize, typer.Option(help="Model size tier.")] = ModelSize.small,
    accelerator: Annotated[str, typer.Option(help="Accelerator: cpu, gpu, or mps.")] = "cpu",
    split: Annotated[str, typer.Option(help="Dataset split to evaluate on.")] = "test",
    predictions_path: Annotated[
        str | None,
        typer.Option(help="Directory to save predictions. Default: <output_path>/<run_id>/eval/predictions."),
    ] = None,
    export_vtk: bool = typer.Option(
        False,
        "--export-vtk",
        help="Export all predictions as VTK point clouds.",
    ),
    compute_forces: bool = typer.Option(
        False,
        "--compute-forces",
        help="Compute Cd/Cl errors during inference (logged to tracker) and save per-sample forces.csv.",
    ),
    query_inference: bool = typer.Option(
        False,
        "--query-inference",
        help="Dense query-based inference for higher-resolution predictions.",
    ),
    num_inference_surface_points: Annotated[
        int, typer.Option(help="Total surface points for query inference.")
    ] = 10000,
    num_inference_volume_points: Annotated[int, typer.Option(help="Total volume points for query inference.")] = 10000,
    query_chunk_size: Annotated[
        int | None,
        typer.Option(help="Query points per chunk per domain. Default: num_surface_anchor_points from model size."),
    ] = None,
    measure_inference_time: bool = typer.Option(
        False,
        "--measure-inference-time",
        help="Record per-sample model inference time and log mean/std/median/min/max. "
        "Useful when sweeping --num-inference-surface-points / --num-inference-volume-points.",
    ),
    precision: Annotated[
        str,
        typer.Option(
            help="Inference precision: float32, float16, or bfloat16. bfloat16 enables Flash "
            "Attention + Tensor Cores on H100/A100 (≈10-15× faster attention).",
        ),
    ] = "float32",
) -> None:
    """Evaluate a trained AB-UPT model and save predictions.

    Computes MSE/MAE/L2 metrics on the test split and saves denormalized
    predictions per sample.  Use ``--export-vtk`` to produce VTP point clouds
    for ParaView visualization.

    With ``--query-inference`` the model predicts at training-resolution anchors
    plus additional query points (processed in chunks), producing denser outputs.

    ``--compute-forces`` computes ground-truth and predicted Cd/Cl using dataset
    surface fields and saved predictions.  Requires ``surface_area_vtp.pt``
    precomputed from the original VTP mesh.
    """
    from aero_cfd.callbacks import AeroMetricsCallbackConfig, QueryInferenceCallbackConfig

    from noether.inference.runners.inference_runner import InferenceRunner

    run_dir = Path(output_path) / run_id
    if predictions_path is None:
        predictions_path = str(run_dir / "eval" / "predictions")

    size_config = MODEL_SIZES[model_size.value]
    preset = _get_preset()
    preset.pipeline_model_overrides[ABUPT_MODEL_KIND].update(size_config.pipeline_overrides)

    forward_props = preset.forward_properties(ABUPT_MODEL_KIND)
    batch_props = list(DOMAIN_POSITION_KEYS.values())

    # --- Build callback + config ---
    extra_datasets = {}

    if query_inference:
        num_sa = size_config.pipeline_overrides.get("num_surface_anchor_points", 512)
        num_va = size_config.pipeline_overrides.get("num_volume_anchor_points", 512)
        if num_inference_surface_points <= num_sa or num_inference_volume_points <= num_va:
            console.print(
                "[red]--query-inference needs strictly more points than anchors in both domains "
                "so that each domain gets at least one query point "
                f"(surface anchors={num_sa}, volume anchors={num_va}), but got "
                f"num_inference_surface_points={num_inference_surface_points}, "
                f"num_inference_volume_points={num_inference_volume_points}.[/red]\n"
                f"[yellow]For model-size '{model_size.value}', choose surface N > {num_sa} "
                f"AND volume N > {num_va}. When sweeping both together, start around "
                f"N ≥ {max(num_sa, num_va) + 8000}.[/yellow]"
            )
            raise typer.Exit(1)
        if query_chunk_size is None:
            query_chunk_size = num_sa

        query_key = f"query_{split}"
        extra_datasets[query_key] = preset.build_dataset(
            split=split,
            root=dataset_root,
            model_kind=ABUPT_MODEL_KIND,
            num_surface_anchor_points=num_inference_surface_points,
            num_volume_anchor_points=num_inference_volume_points,
            use_surface_position_as_input=True,
        )
        callback = QueryInferenceCallbackConfig(
            dataset_key=query_key,
            every_n_epochs=1,
            forward_properties=forward_props,
            save_predictions=True,
            predictions_path=predictions_path,
            batch_properties_to_save=batch_props,
            num_surface_anchors=num_sa,
            num_volume_anchors=num_va,
            query_chunk_size=query_chunk_size,
            compute_forces=compute_forces,
            measure_inference_time=measure_inference_time,
        )
        console.print(f"[bold]Evaluating AB-UPT ({model_size.value}) on DrivAerML/{split} — query inference[/bold]")
        console.print(f"\tRun: {run_id}  Checkpoint: {checkpoint}")
        console.print(f"\tAnchors: {num_sa} surface, {num_va} volume")
        console.print(
            f"\tQueries: {num_inference_surface_points - num_sa} surface, "
            f"{num_inference_volume_points - num_va} volume (chunk={query_chunk_size})"
        )
    else:
        callback = AeroMetricsCallbackConfig(
            dataset_key=split,
            every_n_epochs=1,
            forward_properties=forward_props,
            save_predictions=True,
            predictions_path=predictions_path,
            batch_properties_to_save=batch_props,
            compute_forces=compute_forces,
            measure_inference_time=measure_inference_time,
        )
        console.print(f"[bold]Evaluating AB-UPT ({model_size.value}) on DrivAerML/{split}[/bold]")
        console.print(f"\tRun: {run_id}  Checkpoint: {checkpoint}")

    config = _build_eval_config(
        preset=preset,
        size_config=size_config,
        dataset_root=dataset_root,
        output_path=output_path,
        split=split,
        accelerator=accelerator,
        run_id=run_id,
        checkpoint=checkpoint,
        callbacks=[callback],
        extra_datasets=extra_datasets,
        precision=precision,
    )

    if compute_forces:
        # Include surface_normals and surface_area in the batch for force computation.
        for ds_config in config.datasets.values():
            if ds_config.excluded_properties:
                ds_config.excluded_properties -= {"surface_normals", "surface_area"}

    InferenceRunner.main(device=_get_device(accelerator), config=config)
    console.print(f"\tPredictions saved to: {predictions_path}")

    # --- Post-inference steps ---
    if export_vtk:
        from showcase.utils.vtk_export import export_all_samples, is_pyvista_available

        if not is_pyvista_available():
            console.print("[red]pyvista is required for VTK export. Install with: pip install pyvista[/red]")
            raise typer.Exit(1)
        vtk_dir = export_all_samples(predictions_path, DOMAIN_POSITION_KEYS)
        console.print(f"[green]Exported VTK point clouds to {vtk_dir}[/green]")

    if compute_forces:
        from showcase.utils.forces import compute_forces_for_split

        rows, csv_path = compute_forces_for_split(dataset_root, split, predictions_path)

        skipped_rows = [r for r in rows if "skipped" in r]
        completed_rows = [r for r in rows if "skipped" not in r]
        rows_with_predictions = [r for r in completed_rows if r.get("pred_cd", "") != ""]

        if skipped_rows:
            console.print(f"[yellow]Skipped {len(skipped_rows)} samples (missing files)[/yellow]")
        if completed_rows:
            console.print(f"[green]Saved {csv_path}[/green]")
        if rows_with_predictions:
            mean_drag_error = sum(float(r["drag_error"]) for r in rows_with_predictions) / len(rows_with_predictions)
            mean_lift_error = sum(float(r["lift_error"]) for r in rows_with_predictions) / len(rows_with_predictions)
            console.print(f"\tSamples with predictions: {len(rows_with_predictions)} / {len(completed_rows)}")
            console.print(f"\tMean |dCd| = {mean_drag_error:.4f},  Mean |dCl| = {mean_lift_error:.4f}")
        elif completed_rows:
            console.print(f"\t{len(completed_rows)} ground-truth samples computed (no predictions found).")


# ---------------------------------------------------------------------------
# export-vtk (standalone)
# ---------------------------------------------------------------------------


@app.command(name="export-vtk")
def export_vtk_cmd(
    predictions_path: Annotated[str, typer.Option(help="Path to a saved .pt predictions file.")],
    output_path: Annotated[str, typer.Option(help="Path to write the output VTP file.")],
    domain: Annotated[str, typer.Option(help="Domain to export: 'surface' or 'volume'.")] = "surface",
) -> None:
    """Export a single saved predictions file as a VTP point cloud for ParaView."""
    from showcase.utils.vtk_export import export_pointcloud_to_vtk, is_pyvista_available

    if not is_pyvista_available():
        console.print("[red]pyvista is required for VTK export. Install with: uv pip install pyvista[/red]")
        raise typer.Exit(1)

    data = torch.load(predictions_path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        console.print("[red]Predictions file must contain a dict[str, Tensor].[/red]")
        raise typer.Exit(1)

    pos_key = DOMAIN_POSITION_KEYS.get(domain)
    if pos_key is None or pos_key not in data:
        console.print(f"[red]Position key '{pos_key}' not found. Available: {list(data.keys())}[/red]")
        raise typer.Exit(1)

    positions = data[pos_key]
    fields = {k: v for k, v in data.items() if k.startswith(f"{domain}_") and k != pos_key}

    out = export_pointcloud_to_vtk(positions=positions, fields=fields, output_path=output_path)
    console.print(f"[green]Exported {domain} point cloud ({positions.shape[0]} points) to {out}[/green]")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
