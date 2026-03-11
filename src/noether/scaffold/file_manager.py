#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from importlib.resources.abc import Traversable
from pathlib import Path

from .choices import HardwareChoice, ModelChoice
from .config import TEMPLATES, ScaffoldConfig, substitute


class FileManager:
    """Manages file operations for project scaffolding."""

    @staticmethod
    def _write(path: Path, content: str) -> None:
        """Write *content* to *path*, creating parent directories as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    @staticmethod
    def _copy_template_with_substitution(
        template_file: Traversable, destination_path: Path, config: ScaffoldConfig
    ) -> None:
        """Copy a template file with placeholder substitution."""
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        content = template_file.read_text()
        destination_path.write_text(substitute(content, config))

    @staticmethod
    def _copy_verbatim(template_file: Traversable, destination_path: Path) -> None:
        """Copy a template file verbatim (no substitution)."""
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text(template_file.read_text())

    @staticmethod
    def copy_python_files(config: ScaffoldConfig) -> None:
        """Copy template Python files into the new project with substitutions."""
        tpl = TEMPLATES
        project_dir = config.project_dir
        _copy = FileManager._copy_template_with_substitution
        _verbatim = FileManager._copy_verbatim

        # --- Model files (model-specific) ---
        _copy(tpl / "model" / "base.py", project_dir / "model" / "base.py", config)
        _copy(
            tpl / "schemas" / "models" / "base_config.py",
            project_dir / "schemas" / "models" / "base_config.py",
            config,
        )
        _copy(
            tpl / "model" / f"{config.model.module_name}.py",
            project_dir / "model" / f"{config.model.module_name}.py",
            config,
        )
        _copy(
            tpl / "schemas" / "models" / f"{config.model.schema_module}.py",
            project_dir / "schemas" / "models" / f"{config.model.schema_module}.py",
            config,
        )

        # --- Infrastructure files (with __PROJECT__ substitution) ---
        _copy(tpl / "pipeline" / "__init__.py", project_dir / "pipeline" / "__init__.py", config)
        _copy(
            tpl / "pipeline" / "collators" / "__init__.py",
            project_dir / "pipeline" / "collators" / "__init__.py",
            config,
        )
        _copy(
            tpl / "pipeline" / "collators" / "sparse_tensor_offset.py",
            project_dir / "pipeline" / "collators" / "sparse_tensor_offset.py",
            config,
        )
        _copy(
            tpl / "pipeline" / "multistage_pipelines" / "__init__.py",
            project_dir / "pipeline" / "multistage_pipelines" / "__init__.py",
            config,
        )
        _copy(
            tpl / "pipeline" / "multistage_pipelines" / "aero_multistage.py",
            project_dir / "pipeline" / "multistage_pipelines" / "aero_multistage.py",
            config,
        )
        _copy(
            tpl / "pipeline" / "sample_processors" / "__init__.py",
            project_dir / "pipeline" / "sample_processors" / "__init__.py",
            config,
        )
        _copy(
            tpl / "pipeline" / "sample_processors" / "anchor_point_sampling.py",
            project_dir / "pipeline" / "sample_processors" / "anchor_point_sampling.py",
            config,
        )
        _copy(
            tpl / "trainers" / "automotive_aerodynamics_cfd.py",
            project_dir / "trainers" / "automotive_aerodynamics_cfd.py",
            config,
        )
        _copy(
            tpl / "callbacks" / "surface_volume_evaluation_metrics.py",
            project_dir / "callbacks" / "surface_volume_evaluation_metrics.py",
            config,
        )
        _copy(
            tpl / "schemas" / "datasets" / "aero_dataset_config.py",
            project_dir / "schemas" / "datasets" / "aero_dataset_config.py",
            config,
        )
        _copy(
            tpl / "schemas" / "pipelines" / "aero_pipeline_config.py",
            project_dir / "schemas" / "pipelines" / "aero_pipeline_config.py",
            config,
        )
        _copy(
            tpl / "schemas" / "trainers" / "automotive_aerodynamics_trainer_config.py",
            project_dir / "schemas" / "trainers" / "automotive_aerodynamics_trainer_config.py",
            config,
        )
        _copy(
            tpl / "schemas" / "callbacks" / "callback_config.py",
            project_dir / "schemas" / "callbacks" / "callback_config.py",
            config,
        )
        _copy(tpl / "schemas" / "config_schema.py", project_dir / "schemas" / "config_schema.py", config)

        # --- Static init files (verbatim copies) ---
        _verbatim(tpl / "callbacks" / "__init__.py", project_dir / "callbacks" / "__init__.py")
        _verbatim(tpl / "trainers" / "__init__.py", project_dir / "trainers" / "__init__.py")
        _verbatim(tpl / "schemas" / "__init__.py", project_dir / "schemas" / "__init__.py")
        _verbatim(tpl / "schemas" / "datasets" / "__init__.py", project_dir / "schemas" / "datasets" / "__init__.py")
        _verbatim(tpl / "schemas" / "pipelines" / "__init__.py", project_dir / "schemas" / "pipelines" / "__init__.py")
        _verbatim(tpl / "schemas" / "trainers" / "__init__.py", project_dir / "schemas" / "trainers" / "__init__.py")
        _verbatim(tpl / "schemas" / "callbacks" / "__init__.py", project_dir / "schemas" / "callbacks" / "__init__.py")

    @staticmethod
    def generate_python_files(config: ScaffoldConfig) -> None:
        """Generate dynamic Python files that depend on model choice."""
        proj = config.project_dir
        _write = FileManager._write

        # --- Empty __init__.py files ---
        _write(proj / "__init__.py", "")
        _write(proj / "configs" / "__init__.py", "")

        # --- schemas/models/any_model_config.py (depends on model choice) ---
        cfg_cls = config.model.config_class_name
        schema_mod = config.model.schema_module
        _write(
            proj / "schemas" / "models" / "any_model_config.py",
            f"from typing import Union\n\nfrom .{schema_mod} import {cfg_cls}\n\nAnyModelConfig = Union[{cfg_cls}]\n",
        )

        # --- schemas/models/__init__.py (depends on model choice) ---
        _write(
            proj / "schemas" / "models" / "__init__.py",
            f"from .{config.model.schema_module} import {config.model.config_class_name}\n",
        )

        # --- model/__init__.py (depends on model choice) ---
        _write(
            proj / "model" / "__init__.py",
            f"from .{config.model.module_name} import {config.model.class_name}\n",
        )

    @staticmethod
    def copy_yaml_configs(config: ScaffoldConfig) -> None:
        """Copy all YAML config files into the new project."""
        tpl = TEMPLATES / "configs"
        dst = config.project_dir / "configs"
        ref = config.reference
        _copy = FileManager._copy_template_with_substitution
        _verbatim = FileManager._copy_verbatim

        # --- Verbatim copies (data_specs, normalizers, statistics, datasets, optimizer) ---
        verbatim = [
            ("data_specs", ref.get("data_specs_file")),
            ("dataset_normalizers", ref.get("normalizers_file")),
            ("dataset_statistics", ref.get("statistics_file")),
            ("datasets", ref.get("dataset_config_file")),
        ]
        for subdir, filename in verbatim:
            if filename:
                _verbatim(tpl / subdir / f"{filename}.yaml", dst / subdir / f"{filename}.yaml")

        _verbatim(
            tpl / "optimizer" / f"{config.optimizer.value}.yaml", dst / "optimizer" / f"{config.optimizer.value}.yaml"
        )

        # --- With substitution (model, pipeline, trainer, callbacks, tracker, train) ---
        _copy(tpl / "model" / f"{config.model.value}.yaml", dst / "model" / f"{config.model.value}.yaml", config)

        pipeline_file = ref.get("pipeline_file")
        if pipeline_file:
            _copy(tpl / "pipeline" / f"{pipeline_file}.yaml", dst / "pipeline" / f"{pipeline_file}.yaml", config)

        trainer_file = ref.get("trainer_config_file")
        if trainer_file:
            _copy(tpl / "trainer" / f"{trainer_file}.yaml", dst / "trainer" / f"{trainer_file}.yaml", config)

        callbacks_file = ref.get("callbacks_file")
        if callbacks_file:
            _copy(tpl / "callbacks" / f"{callbacks_file}.yaml", dst / "callbacks" / f"{callbacks_file}.yaml", config)

        _copy(
            tpl / "tracker" / f"{config.tracker.value}.yaml",
            dst / "tracker" / f"{config.tracker.value}.yaml",
            config,
        )

        # --- Train YAML (per-dataset template) ---
        _copy(tpl / f"train_{config.dataset.value}.yaml", dst / "train.yaml", config)

        # Append accelerator for non-GPU hardware
        if config.hardware != HardwareChoice.GPU:
            train_path = dst / "train.yaml"
            content = train_path.read_text()
            train_path.write_text(content + f"accelerator: {config.hardware.value}\n")

        # --- Experiment configs (all 4 models for the dataset's category) ---
        category = ref.get("experiment_category", "shapenet")
        for model in ModelChoice:
            _copy(
                tpl / "experiment" / category / f"{model.value}.yaml",
                dst / "experiment" / f"{model.value}.yaml",
                config,
            )
