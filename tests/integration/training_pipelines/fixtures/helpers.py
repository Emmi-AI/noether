#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir

from noether.core.factory import class_constructor_from_class_path
from noether.core.schemas.schema import ConfigSchema
from noether.training.runners import HydraRunner

from .overrides import apply_test_overrides, to_runner_dict


def make_run_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Create ``tmp_path/out`` and ``tmp_path/data`` and return both."""
    output_path = tmp_path / "out"
    dataset_root = tmp_path / "data"
    output_path.mkdir()
    dataset_root.mkdir()
    return output_path, dataset_root


def instantiate_schema(runner_dict: dict[str, Any]) -> ConfigSchema:
    """Build a ``ConfigSchema`` (or recipe-defined subclass) from a resolved Hydra dict."""
    schema_kind = runner_dict.get("config_schema_kind")
    schema_cls: type[ConfigSchema] = class_constructor_from_class_path(schema_kind) if schema_kind else ConfigSchema
    return schema_cls(**runner_dict)


def compose_recipe_config(
    *,
    configs_dir: str,
    config_name: str,
    overrides: list[str],
    stub_kind: str,
    accelerator: str,
    output_path: Path,
    dataset_root: Path,
    extra: dict[str, Any] | None = None,
) -> ConfigSchema:
    """Compose a recipe YAML, apply test caps, and return a validated ``ConfigSchema``.

    Tests that need to inspect the composed config before training (for assertions
    or further patching) should use this rather than ``run_hydra_recipe`` directly.
    """
    with initialize_config_dir(version_base=None, config_dir=configs_dir, job_name="test"):
        cfg = compose(config_name=config_name, overrides=[*overrides, "tracker=disabled"])

    apply_test_overrides(
        cfg,
        accelerator=accelerator,
        output_path=output_path,
        dataset_root=dataset_root,
        stub_dataset_kind=stub_kind,
        extra=extra,
    )
    return instantiate_schema(to_runner_dict(cfg))


def floating_state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot all floating-point parameters on CPU.

    ``HydraRunner.setup_experiment`` returns a CPU model; ``trainer.train`` moves
    it to ``self.device``. Snapshotting on CPU keeps the post-train comparison
    device-agnostic.
    """
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if v.is_floating_point()}


def assert_any_param_changed(model: torch.nn.Module, before_state: dict[str, torch.Tensor], *, label: str) -> None:
    after = model.state_dict()
    changed = any(not torch.equal(before_state[k], after[k].detach().cpu()) for k in before_state)
    assert changed, f"no parameter changed during {label} training"


def train_and_assert_weights_changed(config: ConfigSchema, *, device: str, label: str) -> None:
    """Set up trainer/model from a fully-built config, train one step, assert weights moved."""
    trainer, model, _tracker, _mc = HydraRunner.setup_experiment(device=device, config=config)
    before = floating_state_cpu(model)
    trainer.train(model)
    assert_any_param_changed(model, before, label=label)


def run_hydra_recipe(
    *,
    configs_dir: str,
    config_name: str,
    overrides: list[str],
    stub_kind: str,
    tmp_path: Path,
    accelerator: str,
    device: str,
    label: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """End-to-end: compose recipe → cap → setup_experiment → train → assert."""
    output_path, dataset_root = make_run_dirs(tmp_path)
    config = compose_recipe_config(
        configs_dir=configs_dir,
        config_name=config_name,
        overrides=overrides,
        stub_kind=stub_kind,
        accelerator=accelerator,
        output_path=output_path,
        dataset_root=dataset_root,
        extra=extra,
    )
    train_and_assert_weights_changed(config, device=device, label=label)
