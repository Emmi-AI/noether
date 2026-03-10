#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from pathlib import Path
from typing import Annotated

import typer

from .choices import DatasetChoice, HardwareChoice, ModelChoice, OptimizerChoice, TrackerChoice
from .config import ScaffoldConfig, resolve_config
from .generator import generate_project

app = typer.Typer(
    name="noether-init",
    help="Scaffold a new Noether training project.",
    add_completion=False,
)


@app.command()
def main(
    project_name: Annotated[str, typer.Argument(help="Project name (valid Python identifier)")],
    model: Annotated[ModelChoice, typer.Option("--model", "-m", help="Model architecture")] = ...,  # type: ignore[assignment]
    dataset: Annotated[DatasetChoice, typer.Option("--dataset", "-d", help="Dataset")] = ...,  # type: ignore[assignment]
    dataset_path: Annotated[str, typer.Option("--dataset-path", help="Path to dataset")] = ...,  # type: ignore[assignment]
    optimizer: Annotated[OptimizerChoice, typer.Option("--optimizer", "-o", help="Optimizer")] = OptimizerChoice.ADAMW,
    tracker: Annotated[
        TrackerChoice, typer.Option("--tracker", "-t", help="Experiment tracker")
    ] = TrackerChoice.DISABLED,
    hardware: Annotated[HardwareChoice, typer.Option("--hardware", help="Hardware target")] = HardwareChoice.GPU,
    project_dir: Annotated[Path, typer.Option("--project-dir", "-l", help="Where to create project dir")] = Path("."),
    wandb_entity: Annotated[
        str | None, typer.Option("--wandb-entity", help="W&B entity (required if tracker=wandb)")
    ] = None,
) -> None:
    """Scaffold a new Noether training project."""
    # Validate project name
    if not project_name.isidentifier():
        typer.echo(f"Error: '{project_name}' is not a valid Python identifier.", err=True)
        raise typer.Exit(1)

    # Validate if wandb has entity set
    if tracker == TrackerChoice.WANDB and not wandb_entity:
        typer.echo("Error: --wandb-entity is required when --tracker=wandb.", err=True)
        raise typer.Exit(1)

    # Resolve to absolute path
    project_dir = (project_dir / project_name).resolve()

    # Check if project dir already exists
    if project_dir.exists():
        typer.echo(f"Error: Directory already exists: {project_dir}", err=True)
        raise typer.Exit(1)

    # Build config
    config = resolve_config(
        project_name=project_name,
        model=model,
        dataset=dataset,
        dataset_path=dataset_path,
        optimizer=optimizer,
        tracker=tracker,
        hardware=hardware,
        project_dir=project_dir,
        wandb_entity=wandb_entity,
    )

    # Generate
    typer.echo(f"Creating project '{project_name}' at {project_dir}")
    generate_project(config)

    # Print summary
    _print_summary(config)


def _print_summary(config: ScaffoldConfig) -> None:
    typer.echo("")
    typer.echo("Project created successfully!")
    typer.echo("")
    typer.echo("Configuration:")
    typer.echo(f"  Project:   {config.project_name}")
    typer.echo(f"  Model:     {config.model.value}")
    typer.echo(f"  Dataset:   {config.dataset.value}")
    typer.echo(f"  Optimizer: {config.optimizer.value}")
    typer.echo(f"  Tracker:   {config.tracker.value}")
    typer.echo(f"  Hardware:  {config.hardware.value}")
    typer.echo(f"  Path:      {config.project_dir}")
    typer.echo("")

    # Suggest run command
    typer.echo("To train, run:")
    typer.echo(f"  uv run noether-train --config-dir {config.project_dir}/configs \\")
    typer.echo(f"    --config-name train +experiment={config.model.value}")
    typer.echo("")
    typer.echo("Experiment configs for all models are in configs/experiment/.")
    typer.echo("")


if __name__ == "__main__":
    app()
