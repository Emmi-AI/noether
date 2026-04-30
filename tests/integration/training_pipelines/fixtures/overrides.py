#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def apply_test_overrides(
    cfg: DictConfig,
    *,
    accelerator: str,
    output_path: Path,
    dataset_root: Path,
    stub_dataset_kind: str,
    keep_datasets: tuple[str, ...] = ("train",),
    extra: dict[str, Any] | None = None,
) -> DictConfig:
    """Cap a composed recipe config to a fast, hermetic single-step run.

    Args:
        cfg: Composed Hydra config (a DictConfig from ``hydra.compose``).
        accelerator: ``"cpu"`` or ``"gpu"``.
        output_path: Directory under which the run writes logs/checkpoints. Tests should pass a ``tmp_path``
            so cleanup is automatic.
        dataset_root: Directory passed as the dataset root. Stub datasets ignore this for I/O but the pydantic schema
            requires the path to exist, so tests should pass ``tmp_path``.
        stub_dataset_kind: Dotted class path of the in-memory stub dataset.
        keep_datasets: Dataset keys to retain. The default ``("train",)`` drops ``val``/``test``/``test_repeat`` etc.;
            those splits only exist to feed callbacks, which we disable below.
        extra: Optional flat mapping of dotted-path → value overrides applied last (e.g.
            ``{"trainer.max_batch_size": 1}``).

    Returns:
        The same ``cfg`` with overrides applied in place (for chaining).
    """
    OmegaConf.set_struct(cfg, False)

    cfg.dataset_root = str(dataset_root)
    cfg.dataset_kind = stub_dataset_kind

    cfg.accelerator = accelerator
    cfg.output_path = str(output_path)
    cfg.store_code_in_output = False
    cfg.tracker = None
    cfg.num_workers = 0
    cfg.devices = "0" if accelerator == "gpu" else "cpu"
    # Recipes ship slurm sections for cluster runs; tests don't need them and some recipes use field names that drift
    # from noether's current SlurmConfig schema. Drop it entirely.
    cfg.slurm = None

    cfg.trainer.max_epochs = 1
    cfg.trainer.max_updates = None
    cfg.trainer.max_samples = None
    cfg.trainer.effective_batch_size = 1
    cfg.trainer.callbacks = []
    cfg.trainer.add_default_callbacks = False
    cfg.trainer.add_trainer_callbacks = False
    cfg.trainer.precision = "float32"
    cfg.trainer.log_every_n_epochs = 1

    cfg.datasets = {k: v for k, v in cfg.datasets.items() if k in keep_datasets}

    if extra:
        for path, value in extra.items():
            OmegaConf.update(cfg, path, value, merge=False)

    return cfg


def to_runner_dict(cfg: DictConfig) -> dict[str, Any]:
    """Resolve a DictConfig and convert to the plain dict ``HydraRunner.run`` expects."""
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise TypeError(f"Expected a dict-like config, got {type(container).__name__}")
    return container  # type: ignore[return-value]
