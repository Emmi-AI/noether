#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from .config import ScaffoldConfig
from .file_manager import FileManager


def generate_project(config: ScaffoldConfig) -> None:
    """Orchestrate full project generation."""
    FileManager.copy_python_files(config)
    FileManager.generate_python_files(config)
    FileManager.copy_yaml_configs(config)
