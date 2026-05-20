#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml
from pydantic import TypeAdapter

from noether.core.callbacks.checkpoint import BestCheckpointCallbackConfig, CheckpointCallbackConfig, EmaCallbackConfig
from noether.core.optimizer.schemas import AnyOptimizerConfig, OptimizerConfig
from noether.core.presets.model_defaults import MODEL_DEFAULTS
from noether.core.schedules.linear_warmup_cosine_decay import LinearWarmupCosineDecayScheduleConfig
from noether.core.schemas import ConfigSchema
from noether.core.schemas.lib import resolve_config_class
from noether.data.base.dataset import DatasetBaseConfig
from noether.data.base.wrappers import DatasetWrappers
from noether.data.preprocessors.normalizers import AnyNormalizer, FieldNormalizerConfig
from noether.training.callbacks import OfflineLossCallbackConfig

logger = logging.getLogger(__name__)


_OPTIMIZER_CONFIG_ADAPTER: TypeAdapter[AnyOptimizerConfig] = TypeAdapter(AnyOptimizerConfig)

CHECKPOINT_CALLBACK = "noether.core.callbacks.CheckpointCallback"
BEST_CHECKPOINT_CALLBACK = "noether.core.callbacks.BestCheckpointCallback"
EMA_CALLBACK = "noether.core.callbacks.EmaCallback"
OFFLINE_LOSS_CALLBACK = "noether.training.callbacks.OfflineLossCallback"
LR_SCHEDULE_LINEAR_WARMUP_COSINE = "noether.core.schedules.LinearWarmupCosineDecaySchedule"
OPTIMIZER_LION = "noether.core.optimizer.Lion"


class DomainPreset(ABC):
    """Base class for domain-specific configuration presets.

    A preset encapsulates domain knowledge - data specs, dataset statistics, normalizer conventions, pipeline defaults
    - so that training scripts only specify what's unique to the experiment (model architecture, hyperparameters,
    dataset path).

    Subclasses must define:
        - ``data_specs``: property returning the domain's data specification object
        - ``dataset_statistics``: property returning pre-computed dataset statistics
        - ``normalizer_spec``: property returning a declarative normalizer mapping
        - ``excluded_properties``: properties to exclude from dataset loading
        - ``target_properties``: list of target property names for this domain

    Subclasses should also set class attributes:
        - ``dataset_kind``: fully qualified dataset class path
        - ``stats``: raw statistics dict
        - ``pipeline_defaults``: default pipeline parameters
        - ``pipeline_model_overrides``: per-model pipeline parameter overrides
        - ``forward_properties_map``: per-model forward property lists (with ``"_default"`` fallback)

    The ``build_normalizers``, ``build_pipeline``, ``build_dataset``, ``build_model``, ``forward_properties``,
    ``standard_callbacks``, and ``build_config`` methods have default implementations that can be overridden when
    needed.
    """

    # --- Class attributes (set by subclasses):
    dataset_kind: str = ""
    stats: dict[str, list[float]] = {}
    stats_file: str | None = None
    pipeline_defaults: dict[str, Any] = {}
    pipeline_model_overrides: dict[str, dict[str, Any]] = {}
    forward_properties_map: dict[str, list[str]] = {}

    @property
    @abstractmethod
    def data_specs(self) -> Any:
        """Return the domain's data specification (e.g., ModelDataSpecs)."""

    @property
    def dataset_statistics(self) -> dict[str, list[float] | float]:
        """Return pre-computed dataset statistics as a flat dict.

        Resolution order:
        1. If ``stats_file`` is set on the preset, loads from that YAML file.
        2. If ``stats`` dict is set on the preset, returns a copy.
        3. If ``dataset_kind`` points to a class with a ``STATS_FILE`` attribute, loads from that.

        Subclasses can override this property for custom logic.
        """
        if self.stats_file is not None:
            return self._load_yaml(self.stats_file)
        if self.stats:
            return dict(self.stats)

        # Fall back to STATS_FILE on the dataset class:
        if self.dataset_kind:
            module_name, class_name = self.dataset_kind.rsplit(".", 1)
            module = importlib.import_module(module_name)
            dataset_cls = getattr(module, class_name)
            stats_path = getattr(dataset_cls, "STATS_FILE", None)
            if stats_path is not None:
                return self._load_yaml(stats_path)

        raise ValueError(
            f"{type(self).__name__}: no stats available. Set 'stats', 'stats_file', "
            "or ensure the dataset class has a STATS_FILE attribute."
        )

    @property
    @abstractmethod
    def normalizer_spec(self) -> dict[str, FieldNormalizerConfig]:
        """Declarative normalizer mapping.

        Keys are data source names (e.g., ``"surface_pressure"``).
        Values are ``FieldNormalizerConfig`` instances that declare the normalization strategy.
        Statistics are resolved at runtime by the dataset from its ``STATS_FILE``.

        Example::

            {
                "surface_pressure": FieldNormalizerConfig(strategy="mean_std"),
                "surface_position": FieldNormalizerConfig(strategy="position", scale=1000),
                "volume_vorticity": FieldNormalizerConfig(
                    strategy="mean_std",
                    logscale=True,
                    stat_keys={"mean": "volume_vorticity_logscale_mean", "std": "volume_vorticity_logscale_std"},
                ),
            }
        """

    @property
    @abstractmethod
    def excluded_properties(self) -> set[str] | None:
        """Properties to exclude from dataset loading, or None."""

    @abstractmethod
    def target_properties(self) -> list[str]:
        """Return the list of target property names for this domain."""

    def forward_properties(self, model_kind: str) -> list[str]:
        """Return the list of forward properties for the given model architecture.

        Looks up ``forward_properties_map`` by model kind, falling back to ``"_default"``.
        """
        if model_kind in self.forward_properties_map:
            return list(self.forward_properties_map[model_kind])
        return list(self.forward_properties_map.get("_default", []))

    def build_pipeline(self, model_kind: str, **overrides: Any) -> Any:
        """Build a pipeline config by merging defaults, model overrides, and user overrides.

        Subclasses must override this to construct the appropriate pipeline config.
        The default implementation merges ``pipeline_defaults``, model-specific overrides from
        ``pipeline_model_overrides``, and any caller-provided overrides into a single dict and returns it.
        Subclasses should call ``super()`` to get the merged params.
        """
        params = {**self.pipeline_defaults}
        if model_kind in self.pipeline_model_overrides:
            params.update(self.pipeline_model_overrides[model_kind])
        params.update(overrides)
        return params

    @abstractmethod
    def build_dataset(
        self,
        *,
        split: str,
        root: str,
        model_kind: str,
        wrappers: list[DatasetWrappers] | None = None,
        **overrides: Any,
    ) -> DatasetBaseConfig:
        """Build a dataset config for the given split."""

    def build_normalizers(self) -> dict[str, list[AnyNormalizer]]:
        """Build normalizer configs from the declarative ``normalizer_spec``.

        Wraps each ``FieldNormalizerConfig`` from the spec into a single-element list,
        matching the format expected by ``DatasetBaseConfig.dataset_normalizers``.

        Returns:
            Dict mapping data source names to lists of normalizer configs.
        """
        return {key: [config] for key, config in self.normalizer_spec.items()}

    @staticmethod
    def standard_callbacks(
        *,
        log_every_n_epochs: int = 1,
        save_every_n_epochs: int = 10,
        eval_dataset_key: str = "test",
        batch_size: int = 1,
        ema: bool = True,
        ema_factors: set[float] | None = None,
        best_metric_key: str = "loss/test/total",
    ) -> list:
        """Build a standard set of training callbacks.

        Returns a list containing:
            - CheckpointCallback (periodic checkpoints)
            - OfflineLossCallback (validation loss)
            - BestCheckpointCallback (saves best model by metric)
            - EmaCallback (optional exponential moving average)

        Args:
            log_every_n_epochs: frequency for loss logging and validation.
            save_every_n_epochs: frequency for checkpoint saving and EMA.
            eval_dataset_key: dataset key for offline evaluation.
            batch_size: batch size for evaluation callbacks.
            ema: whether to include EMA callback.
            ema_factors: EMA decay factors. Defaults to None, numerically it will be {0.9999}.
            best_metric_key: metric key for best checkpoint selection.
        """
        callbacks: list = [
            CheckpointCallbackConfig(
                kind=CHECKPOINT_CALLBACK,
                every_n_epochs=save_every_n_epochs,
                save_weights=True,
                save_latest_weights=True,
            ),
            OfflineLossCallbackConfig(
                kind=OFFLINE_LOSS_CALLBACK,
                every_n_epochs=log_every_n_epochs,
                dataset_key=eval_dataset_key,
                batch_size=batch_size,
            ),
            BestCheckpointCallbackConfig(
                kind=BEST_CHECKPOINT_CALLBACK,
                every_n_epochs=log_every_n_epochs,
                metric_key=best_metric_key,
            ),
        ]
        if ema:
            callbacks.append(
                EmaCallbackConfig(
                    kind=EMA_CALLBACK,
                    every_n_epochs=save_every_n_epochs,
                    target_factors=list(ema_factors or {0.9999}),
                    save_weights=False,
                    save_last_weights=False,
                    save_latest_weights=True,
                )
            )
        return callbacks

    def build_optimizer(
        self,
        *,
        kind: str = OPTIMIZER_LION,
        lr: float = 5e-5,
        weight_decay: float = 0.05,
        clip_grad_norm: float | None = 1.0,
        warmup_percent: float = 0.05,
        end_lr: float | None = 1e-6,
    ) -> OptimizerConfig:
        """Build an optimizer config with sensible defaults.

        Args:
            kind: optimizer class path.
            lr: learning rate.
            weight_decay: weight decay.
            clip_grad_norm: gradient clipping norm. None to disable.
            warmup_percent: fraction of training for linear warmup.
            end_lr: final learning rate for cosine decay. None to disable scheduling.
        """
        schedule = None
        if end_lr is not None:
            schedule = LinearWarmupCosineDecayScheduleConfig(
                kind=LR_SCHEDULE_LINEAR_WARMUP_COSINE,
                warmup_percent=warmup_percent,
                end_value=end_lr,
                max_value=lr,
            )

        return _OPTIMIZER_CONFIG_ADAPTER.validate_python(
            {
                "kind": kind,
                "lr": lr,
                "weight_decay": weight_decay,
                "clip_grad_norm": clip_grad_norm,
                "schedule_config": schedule,
            }
        )

    def build_model(
        self,
        *,
        model_kind: str,
        optimizer: OptimizerConfig | None = None,
        **model_params: Any,
    ) -> Any:
        """Build a model config from the model kind and parameters.

        Automatically injects ``data_specs``, ``forward_properties``, ``optimizer_config``, and ``kind`` so the user
        only provides architecture knobs.

        If the model kind has registered defaults in ``_MODEL_DEFAULTS``, those are applied before constructing
        the config (e.g., AB-UPT sub-configs).

        Args:
            model_kind: fully qualified class path of the model.
            optimizer: optimizer config. Uses ``build_optimizer()`` defaults if None.
            **model_params: model-specific parameters (e.g., ``hidden_dim``, ``num_heads``).

        Returns:
            A model config object.
        """
        # Apply registered model defaults (e.g., AB-UPT sub-configs):
        if model_kind in MODEL_DEFAULTS:
            MODEL_DEFAULTS[model_kind](self.data_specs, model_params)

        model_config_cls = self._resolve_config_class(
            model_kind,
            base_module="noether.core.schemas.models.base",
            base_class_name="ModelBaseConfig",
        )

        kwargs: dict[str, Any] = {
            "kind": model_kind,
            "data_specs": self.data_specs,
            "forward_properties": self.forward_properties(model_kind),
            "optimizer_config": optimizer or self.build_optimizer(),
            **model_params,
        }
        return model_config_cls(**kwargs)

    def build_config(
        self,
        *,
        model_kind: str,
        model_params: dict[str, Any] | None = None,
        model_config: Any | None = None,
        optimizer: OptimizerConfig | None = None,
        trainer_kind: str,
        trainer_params: dict[str, Any] | None = None,
        dataset_root: str,
        output_path: str | None = None,
        datasets: dict[str, str] | list[str] | None = None,
        extra_datasets: dict[str, DatasetBaseConfig] | None = None,
        callbacks_override: list | None = None,
        extra_callbacks: list | None = None,
        accelerator: str | None = None,
        max_epochs: int = 500,
        batch_size: int = 1,
        seed: int = 42,
        **config_overrides: Any,
    ) -> ConfigSchema:
        """Assemble a complete ConfigSchema with all domain defaults filled in.

        Provide either ``model_config`` (pre-built) or ``model_params`` (dict of architecture knobs like ``hidden_dim``,
        ``num_heads``). If ``model_params`` is used, ``build_model()`` is called automatically.

        Args:
            model_kind: fully qualified class path of the model.
            model_params: model architecture parameters (used with ``build_model``).
            model_config: pre-built model config object. Mutually exclusive with ``model_params``.
            optimizer: optimizer config. Defaults to Lion with cosine decay via ``build_optimizer()``.
            trainer_kind: fully qualified class path of the trainer.
            trainer_params: additional trainer-specific parameters (e.g., loss weights).
            dataset_root: root directory of the dataset.
            output_path: output directory. Defaults to ``{dataset_root}/outputs``.
            datasets: splits to create. Either a list of split names (e.g., ``["train", "test"]``)
                where keys equal splits, or a dict mapping keys to splits for custom naming
                (e.g., ``{"my_train": "train"}``). Defaults to ``["train", "test"]``.
            extra_datasets: additional pre-built dataset configs to merge in (e.g., repeated test sets).
            callbacks_override: replace the default callback list entirely. Defaults to ``standard_callbacks()``.
            extra_callbacks: additional callbacks appended to the default (or overridden) list.
            accelerator: "cpu", "gpu", or "mps". Auto-detected if None.
            max_epochs: maximum training epochs.
            batch_size: effective batch size.
            seed: random seed.
            **config_overrides: additional fields passed to ConfigSchema.

        Returns:
            A fully populated ConfigSchema ready for ``HydraRunner().main()``.
        """
        if model_config is not None and model_params is not None:
            raise ValueError("Provide either 'model_config' or 'model_params', not both.")

        if output_path is None:
            output_path = str(Path(dataset_root) / "outputs")

        if datasets is None:
            datasets = ["train", "test"]
        if isinstance(datasets, list):
            datasets = {s: s for s in datasets}

        callbacks = callbacks_override or self.standard_callbacks(batch_size=batch_size)
        if extra_callbacks:
            callbacks = callbacks + extra_callbacks

        optimizer_config = optimizer

        # Build model config if not provided directly:
        if model_config is None:
            model_config = self.build_model(
                model_kind=model_kind,
                optimizer=optimizer_config,
                **(model_params or {}),
            )

        # Build dataset configs:
        dataset_configs: dict[str, Any] = {}
        for key, split in datasets.items():
            dataset_configs[key] = self.build_dataset(
                split=split,
                root=dataset_root,
                model_kind=model_kind,
            )
        if extra_datasets:
            dataset_configs.update(extra_datasets)

        # Build trainer config:
        forward_props = self.forward_properties(model_kind)
        target_props = self.target_properties()

        trainer_kwargs: dict[str, Any] = {
            "kind": trainer_kind,
            "max_epochs": max_epochs,
            "effective_batch_size": batch_size,
            "log_every_n_epochs": 1,
            "callbacks": callbacks,
            "forward_properties": forward_props,
            "target_properties": target_props,
            **(trainer_params or {}),
        }

        trainer_config_cls = self._resolve_config_class(
            trainer_kind,
            base_module="noether.core.schemas.trainers",
            base_class_name="BaseTrainerConfig",
        )
        trainer_config = trainer_config_cls(**trainer_kwargs)

        # Build ConfigSchema:
        schema_kwargs: dict[str, Any] = {
            "datasets": dataset_configs,
            "model": model_config,
            "trainer": trainer_config,
            "output_path": output_path,
            "seed": seed,
            "dataset_statistics": {
                k: [v] if isinstance(v, (int, float)) else v for k, v in self.dataset_statistics.items()
            },
            **config_overrides,
        }
        if accelerator is not None:
            schema_kwargs["accelerator"] = accelerator

        return ConfigSchema(**schema_kwargs)

    @staticmethod
    def _load_yaml(path: str) -> dict[str, list[float] | float]:
        """Load dataset statistics from a YAML file.

        Args:
            path: path to the YAML file (absolute or relative to cwd).

        Returns:
            Dict mapping stat names to lists of floats (or scalar floats).
        """
        resolved = Path(path).expanduser()
        with open(resolved) as f:
            data = yaml.safe_load(f)
        result: dict[str, list[float] | float] = {}
        for k, v in data.items():
            if isinstance(v, list):
                result[k] = [float(x) for x in v]
            else:
                result[k] = float(v)
        return result

    @staticmethod
    def _resolve_config_class(
        kind: str,
        base_module: str,
        base_class_name: str,
    ) -> type:
        """Resolve a runtime class from its kind string, then find its config class.

        Delegates to :func:`noether.core.schemas.lib.resolve_config_class`.

        Args:
            kind: fully qualified class path (e.g., ``"noether.training.trainers.WeightedLossTrainer"``).
            base_module: module containing the base config class to check against.
            base_class_name: name of the base config class.
        """
        base_mod = importlib.import_module(base_module)
        base_cls = getattr(base_mod, base_class_name)
        return resolve_config_class(kind, base_cls)
