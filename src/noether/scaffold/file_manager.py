#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from .choices import HardwareChoice
from .config import TEMPLATES, ScaffoldConfig, substitute


class FileManager:
    """Manages file operations for project scaffolding."""

    @staticmethod
    def copy_template_tree(config: ScaffoldConfig) -> None:
        """Recursively copy all template files into the project directory with substitution.

        Creates a nested layout:
            project_dir/
                pyproject.toml
                project_name/
                    __init__.py, callbacks/, models/, configs/, ...
        """
        _skip_names = {"README.MD"}
        _allowed_suffixes = {".py", ".yaml", ".toml"}
        _root_files = {"pyproject.toml"}
        package_dir = config.project_dir / config.project_name
        for source in TEMPLATES.rglob("*"):
            if not source.is_file() or source.name in _skip_names or source.suffix not in _allowed_suffixes:
                continue

            rel_path = source.relative_to(TEMPLATES)
            if source.name in _root_files:
                dest = config.project_dir / rel_path
            else:
                dest = package_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            content = source.read_text()
            dest.write_text(substitute(content, config))

        # Append accelerator for non-GPU hardware
        if config.hardware != HardwareChoice.GPU:
            experiment = package_dir / "configs" / "base_experiment.yaml"
            content = experiment.read_text()
            experiment.write_text(content + f"accelerator: {config.hardware.value}\n")
