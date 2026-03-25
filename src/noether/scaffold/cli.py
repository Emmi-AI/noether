#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from pathlib import Path
from typing import Annotated

import typer

from .choices import HardwareChoice, TrackerChoice
from .config import ScaffoldConfig, resolve_config
from .file_manager import FileManager

app = typer.Typer(
    name="noether-init",
    help="Scaffold a new Noether training project.",
    add_completion=False,
)


@app.command()
def main(
    project_name: Annotated[
        str,
        typer.Argument(
            help="Project name (valid Python identifier). Examples: 'my_project', 'MyProject1'). No hyphens allowed."
        ),
    ],
    tracker: Annotated[
        TrackerChoice, typer.Option("--tracker", "-t", help="Experiment tracker")
    ] = TrackerChoice.DISABLED,
    hardware: Annotated[HardwareChoice, typer.Option("--hardware", help="Hardware target")] = HardwareChoice.GPU,
    project_dir: Annotated[Path, typer.Option("--project-dir", "-d", help="Where to create project dir")] = Path("."),
    wandb_entity: Annotated[
        str | None, typer.Option("--wandb-entity", help="W&B entity, e.g. 'my-team' (defaults to your W&B username)")
    ] = None,
) -> None:
    """Scaffold a new Noether training project."""
    # Validate project name
    if not project_name.isidentifier():
        typer.echo(f"Error: '{project_name}' is not a valid Python identifier.", err=True)
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
        tracker=tracker,
        hardware=hardware,
        project_dir=project_dir,
        wandb_entity=wandb_entity,
    )

    # Generate
    typer.echo(f"Creating project '{project_name}' at {project_dir}")
    FileManager.copy_template_tree(config)

    # Print summary
    _print_summary(config)


def _print_summary(config: ScaffoldConfig) -> None:
    typer.echo(
        "\nProject created successfully!\n"
        "Configuration:\n"
        f"  Project:   {config.project_name}\n"
        f"  Tracker:   {config.tracker.value}\n"
        f"  Hardware:  {config.hardware.value}\n"
        f"  Path:      {config.project_dir}\n"
    )
    # Suggest run command
    typer.echo(
        "To train, run:\n"
        f"  cd {config.project_dir}\n"
        f"  uv run noether-train --hp {config.project_name}/configs/base_experiment.yaml\n"
    )


if __name__ == "__main__":
    app()
