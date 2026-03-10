#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from .config import ScaffoldConfig
from .file_copier import copy_python_files, copy_yaml_configs, generate_python_files


def generate_project(config: ScaffoldConfig) -> None:
    """Orchestrate full project generation."""
    copy_python_files(config)
    generate_python_files(config)
    copy_yaml_configs(config)
