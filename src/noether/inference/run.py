#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Notebook-friendly Python API for loading a trained run.

The non-Hydra counterpart to ``noether-eval``: instead of spinning up an
:class:`~noether.inference.runners.InferenceRunner` (with trainer context, callbacks, tracker, etc.), it gives you a
single handle to a run from which you can pull the resolved config, an instantiated dataset, and a model with checkpoint
weights loaded.

Two ways to build a :class:`Run`:

- :class:`Run` (``run_dir``) — open a full training output directory (``hp_resolved.yaml`` + ``checkpoints/``).
  Gives access to the resolved config, the dataset, normalizers, and the model.
- :meth:`Run.from_checkpoint` (``path``) — open just a single ``..._model.th`` file. Every checkpoint written by
  noether's :class:`~noether.core.writers.CheckpointWriter` embeds the model config, the discriminator kind, and the
  per-field normalizer specs + statistics, which is enough for :meth:`model` and :meth:`normalizers` without the run
  directory. :meth:`dataset` and :attr:`config` are unavailable in this mode.

.. code-block:: python

    from noether.inference import Run

    # Full run directory.
    run = Run("/outputs/2026-04-09_abc12")
    for ds in run.config.datasets.values():
        ds.root = "/local/path/to/data"
    dataset = run.dataset("test")
    model = run.model(checkpoint="latest", device="cuda")

    # Single checkpoint file — no run dir, no hp_resolved.yaml, no stats file.
    run = Run.from_checkpoint("/outputs/.../checkpoints/ab_upt_cp=last_model.th")
    model = run.model(device="cuda")
    norms = run.normalizers()

For reproducible eval with metrics, callbacks, and full logging, use ``noether-eval`` instead.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from torch import nn

from noether.core.factory import Factory
from noether.core.factory.dataset import DatasetFactory
from noether.core.factory.utils import class_constructor_from_class_path
from noether.core.schemas.lib import resolve_config_class
from noether.core.schemas.models import ModelBaseConfig
from noether.core.schemas.normalizers import NormalizerConfig
from noether.core.schemas.schema import ConfigSchema
from noether.core.types import CheckpointKeys
from noether.core.utils.model import compute_model_norm
from noether.data.base.dataset import Dataset
from noether.data.preprocessors.compose import ComposePreProcess

__all__ = ["Run", "sanitize_hp_resolved"]


def _to_plain_python(obj: Any) -> Any:
    """Recursively convert tuples/sets to lists so the YAML round-trips through ``yaml.safe_dump``."""
    if isinstance(obj, dict):
        return {k: _to_plain_python(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, set, frozenset)):
        return [_to_plain_python(v) for v in obj]
    if isinstance(obj, list):
        return [_to_plain_python(v) for v in obj]
    return obj


def _load_hp_resolved_as_plain_dict(hp_resolved_path: Path) -> dict:
    """Read ``hp_resolved.yaml`` and coerce tuples/sets to lists so pydantic / Hydra can consume it.

    ``hp_resolved.yaml`` is written with :func:`yaml.dump`, which emits ``!!python/tuple`` tags for tuple values
    (notably ``dataset_statistics``); both Hydra and pydantic prefer plain YAML.
    """
    with open(hp_resolved_path) as f:
        return cast("dict", _to_plain_python(yaml.full_load(f)))


def sanitize_hp_resolved(hp_resolved_path: Path) -> Path:
    """Write a tag-free copy of ``hp_resolved.yaml`` to a temp file.

    Equivalent to :func:`_load_hp_resolved_as_plain_dict` plus a ``yaml.safe_dump`` to a fresh tempdir. Kept for
    external callers; internal loading goes through the in-memory helper to avoid leaking tempdirs.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="noether_eval_"))
    safe_path = tmp_dir / "hp_resolved.yaml"
    with open(safe_path, "w") as f:
        yaml.safe_dump(_load_hp_resolved_as_plain_dict(hp_resolved_path), f, sort_keys=False)
    return safe_path


class Run:
    """Handle to a trained run.

    Two construction modes, picked by which constructor you use:

    - :class:`Run` (``run_dir``) — full run directory: reads ``hp_resolved.yaml`` and validates it against
      :class:`ConfigSchema`. All accessors below are available.
    - :meth:`Run.from_checkpoint` (``path``) — just a single ``..._model.th`` file: reads embedded model config +
      normalizer payload. :meth:`model` and :meth:`normalizers` work; :meth:`dataset`, :attr:`config`, and
      :attr:`statistics` raise.

    Mutate :attr:`config` between construction and the lazy methods to override training-time values (typically dataset
    roots when the run was produced on a different machine). Only meaningful in run-dir mode.

    Args:
        run_dir: Path to the training run output directory (the one that contains ``hp_resolved.yaml`` and a
            ``checkpoints/`` subdirectory). Typically ``output_path/run_id`` or ``output_path/run_id/stage_name``.

    Attributes:
        run_dir: Resolved absolute path to the run directory in run-dir mode; ``None`` in checkpoint-only mode.
        checkpoint_path: Resolved absolute path to the ``.th`` file in checkpoint-only mode; ``None`` in run-dir mode.

    Raises:
        FileNotFoundError: If ``run_dir`` does not exist or doesn't contain ``hp_resolved.yaml``.

    Example:

        .. testcode::
            :skipif: True  # requires a real run directory

            from noether.inference import Run

            # Bring-your-own-data flow: apply the trained model to a custom input dict, then denormalize the predictions.
            run = Run.from_checkpoint("/outputs/.../ab_upt_cp=last_model.th")
            model = run.model(device="cuda")
            norms = run.normalizers()
            with torch.inference_mode():
                pred = model(**my_inputs)
            pred_phys = norms["surface_pressure"].inverse(pred["surface_pressure"])
    """

    def __init__(self, run_dir: Path | str):
        resolved = Path(run_dir).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"run_dir does not exist: {resolved}")
        self.run_dir: Path | None = resolved
        self.checkpoint_path: Path | None = None
        self._checkpoint_data: dict | None = None
        self._config: ConfigSchema | None = self._load_config(resolved)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path | str) -> Run:
        """Build a :class:`Run` from a single ``..._model.th`` file.

        Reads the model config (:attr:`CheckpointKeys.MODEL_CONFIG`), the discriminator kind
        (:attr:`CheckpointKeys.CONFIG_KIND`), and — if present — the per-field normalizer payload
        (:attr:`CheckpointKeys.NORMALIZER_CONFIGS` /
        :attr:`CheckpointKeys.NORMALIZER_STATISTICS`) that :class:`~noether.core.writers.CheckpointWriter` embeds
        in every checkpoint.

        The model class itself must still be importable in the current process — the kind string points at a class,
        not at its implementation. If the checkpoint references a recipe-specific model, make sure that recipe is
        installed (or on :data:`sys.path`) before calling.

        Args:
            checkpoint_path: Path to a ``..._model.th`` file written by noether.

        Returns:
            A :class:`Run` in checkpoint-only mode. :meth:`model` and :meth:`normalizers` are usable;
            :meth:`dataset` and :attr:`config` raise.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            KeyError: If the checkpoint is missing any of ``state_dict``, ``model_config``, or ``config_kind`` (older
                checkpoints predate the embedded config — fall back to ``Run(run_dir)``).
        """
        ckpt_path = Path(checkpoint_path).expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        for required in (CheckpointKeys.STATE_DICT, CheckpointKeys.CONFIG_KIND, CheckpointKeys.MODEL_CONFIG):
            if required not in ckpt_data:
                raise KeyError(
                    f"checkpoint at {ckpt_path} is missing {required!r}. "
                    "Older runs predate the embedded model config — load via Run(run_dir) "
                    "against the run directory instead."
                )

        obj = cls.__new__(cls)
        obj.run_dir = None
        obj.checkpoint_path = ckpt_path
        obj._checkpoint_data = ckpt_data
        obj._config = None
        return obj

    @property
    def is_checkpoint_only(self) -> bool:
        """``True`` if this :class:`Run` was built via :meth:`from_checkpoint` (no run dir, no resolved config)."""
        return self._checkpoint_data is not None

    @property
    def config(self) -> ConfigSchema:
        """Validated :class:`ConfigSchema` loaded from ``hp_resolved.yaml``.

        Safe to mutate before calling :meth:`dataset` / :meth:`model` / :meth:`normalizers`.

        Raises:
            RuntimeError: If this :class:`Run` was built via :meth:`from_checkpoint` — no run directory means no
                resolved config.
        """
        if self._config is None:
            raise RuntimeError(
                "Run.config is unavailable in checkpoint-only mode (built via "
                "Run.from_checkpoint). Open the run directory with Run(run_dir) "
                "if you need to inspect or mutate the training config."
            )
        return self._config

    @config.setter
    def config(self, value: ConfigSchema) -> None:
        self._config = value

    def _load_config(self, run_dir: Path) -> ConfigSchema:
        hp_path = run_dir / "hp_resolved.yaml"
        if not hp_path.exists():
            raise FileNotFoundError(
                f"hp_resolved.yaml not found in {run_dir}. "
                "Make sure run_dir points at a training run output directory "
                "(typically output_path/run_id[/stage_name])."
            )
        data = _load_hp_resolved_as_plain_dict(hp_path)

        # ConfigSchema's _resolve_slurm_defaults validator does
        # ``validate_path(output_path, mkdir=True)`` on whatever path the
        # training run wrote — typically a server path that doesn't make
        # sense on this machine. Anchor the loaded config to the local
        # run_dir so the validator's mkdir is a no-op.
        data["output_path"] = str(run_dir)
        return ConfigSchema(**data)

    @property
    def statistics(self) -> dict[str, list[float | int]]:
        """Training-time dataset statistics (``config.dataset_statistics`` or ``{}``).

        Convenience accessor for the stat values the training run computed — typically per-field means/stds used by the
        trainer's pipeline. Returns an empty dict if the run didn't compute any stats.

        Note: this is separate from the dataset class's static ``STATS_FILE``, which :meth:`normalizers` reads in
        run-dir mode.

        Raises:
            RuntimeError: In checkpoint-only mode (no resolved config).
        """
        if self.is_checkpoint_only:
            raise RuntimeError(
                "Run.statistics is unavailable in checkpoint-only mode (built via "
                "Run.from_checkpoint). Open the run directory with Run(run_dir) "
                "for the training-time dataset statistics."
            )
        if self._config is None:
            raise RuntimeError("Run._config is None in run-dir mode — Run.config may have been cleared.")
        return dict(self._config.dataset_statistics or {})

    def normalizers(self, split: str = "test") -> dict[str, ComposePreProcess]:
        """Build the trained run's field normalizers without instantiating its dataset.

        In **run-dir mode**, reads the dataset class's ``STATS_FILE`` (looked up from ``config.datasets[split].kind``)
        and constructs each normalizer from ``config.datasets[split].dataset_normalizers``. The data root is never
        touched.

        In **checkpoint-only mode**, reads the per-field preprocessor configs and resolved statistics that
        ``CheckpointWriter`` embeds in every checkpoint (``NORMALIZER_CONFIGS`` / ``NORMALIZER_STATISTICS``).
        The ``split`` argument is ignored — only the writer-side split (typically ``test``) was captured.

        Args:
            split: Dataset key to source the normalizer configs from. Splits typically share normalizers;
                the arg is provided for parity with :meth:`dataset`. Ignored in checkpoint-only mode.

        Returns:
            Dict mapping field name (e.g. ``"surface_pressure"``) to a :class:`ComposePreProcess`.
            Empty dict if no normalizers are available for this split.

        Raises:
            KeyError: In run-dir mode, if ``split`` is not in ``self.config.datasets``. In checkpoint-only mode, if the
                checkpoint predates the embedded normalizer keys.
        """
        if self.is_checkpoint_only:
            return self._normalizers_from_checkpoint_data()
        return self._normalizers_from_config(split)

    def _normalizers_from_config(self, split: str) -> dict[str, ComposePreProcess]:
        dataset_config = self._dataset_config(split)
        if not dataset_config.dataset_normalizers:
            return {}

        # Resolve the dataset class only to read its STATS_FILE — never instantiate it.
        dataset_cls = class_constructor_from_class_path(dataset_config.kind)
        stats_path = getattr(dataset_cls, "STATS_FILE", None)
        statistics: dict[str, list[float] | float] | None = None
        if stats_path is not None:
            with open(Path(stats_path).expanduser()) as f:
                raw = yaml.safe_load(f) or {}
            statistics = {k: ([float(x) for x in v] if isinstance(v, list) else float(v)) for k, v in raw.items()}

        return self._build_normalizers(dataset_config.dataset_normalizers, statistics)

    def _normalizers_from_checkpoint_data(self) -> dict[str, ComposePreProcess]:
        if self._checkpoint_data is None:
            raise RuntimeError(
                "Run._normalizers_from_checkpoint_data called outside checkpoint-only mode (_checkpoint_data is None)."
            )
        ckpt = self._checkpoint_data
        if CheckpointKeys.NORMALIZER_CONFIGS not in ckpt:
            raise KeyError(
                f"checkpoint at {self.checkpoint_path} is missing "
                f"{CheckpointKeys.NORMALIZER_CONFIGS!r}. Older runs predate embedded "
                "normalizer info — re-train with the current code, or open the run "
                "via Run(run_dir) and call .normalizers(split) instead."
            )
        configs_dump = ckpt[CheckpointKeys.NORMALIZER_CONFIGS]
        statistics = ckpt.get(CheckpointKeys.NORMALIZER_STATISTICS)

        # Each entry is a plain dict from ``model_dump`` — re-validate back into a
        # pydantic ``NormalizerConfig`` so Factory can read its ``kind`` field.
        validated: dict[str, NormalizerConfig | list[NormalizerConfig]] = {}
        for key, configs in configs_dump.items():
            configs_list = configs if isinstance(configs, list) else [configs]
            validated[key] = [resolve_config_class(c["kind"], NormalizerConfig).model_validate(c) for c in configs_list]
        return self._build_normalizers(validated, statistics)

    def _build_normalizers(
        self,
        configs_by_key: dict[str, NormalizerConfig | list[NormalizerConfig]],
        statistics: dict[str, Any] | None,
    ) -> dict[str, ComposePreProcess]:
        """Instantiate one :class:`ComposePreProcess` per field from validated normalizer configs.

        Shared tail of :meth:`_normalizers_from_config` and :meth:`_normalizers_from_checkpoint_data`. Each value in
        ``configs_by_key`` may be a single config or a list of configs.
        """
        normalizers: dict[str, ComposePreProcess] = {}
        for key, configs in configs_by_key.items():
            configs_list = configs if isinstance(configs, list) else [configs]
            preprocessors = [
                Factory().instantiate(cfg, normalization_key=key, statistics=statistics) for cfg in configs_list
            ]
            normalizers[key] = ComposePreProcess(normalization_key=key, preprocessors=preprocessors)
        return normalizers

    def dataset(self, split: str = "test") -> Dataset:
        """Instantiate the dataset for ``split``.

        Wires up the collator (``dataset.pipeline``) the same way the trainer does, so the dataset can be plugged into a
        :class:`torch.utils.data.DataLoader` for batched forward passes.

        Args:
            split: Dataset key (e.g. ``"train"``, ``"val"``, ``"test"``).

        Raises:
            RuntimeError: In checkpoint-only mode (the checkpoint doesn't know about the original dataset configuration).
            KeyError: If ``split`` is not in ``self.config.datasets``.
        """
        if self.is_checkpoint_only:
            raise RuntimeError(
                "Run.dataset() is unavailable in checkpoint-only mode (built via "
                "Run.from_checkpoint). The checkpoint doesn't carry the dataset "
                "configuration. Open the run directory with Run(run_dir) to "
                "instantiate the trained run's dataset, or build your own tensors "
                "and apply Run.normalizers() / Run.model() to them directly."
            )
        dataset_config = self._dataset_config(split)
        dataset: Dataset = DatasetFactory().create(dataset_config)  # type: ignore[assignment]
        pipeline = Factory().create(dataset_config.pipeline)
        if pipeline is not None:
            dataset.pipeline = pipeline
        return dataset

    def _dataset_config(self, split: str):
        """Look up ``self.config.datasets[split]`` with a friendly KeyError listing available splits."""
        if self._config is None:
            raise RuntimeError("Run._config is None in run-dir mode — Run.config may have been cleared.")
        if split not in self._config.datasets:
            raise KeyError(
                f"split {split!r} not found in config.datasets. "
                f"Available splits: {sorted(self._config.datasets.keys())}"
            )
        return self._config.datasets[split]

    def model(
        self,
        *,
        checkpoint: str = "latest",
        device: str | torch.device = "cpu",
    ) -> nn.Module:
        """Instantiate the model and load checkpoint weights.

        Unlike the training/eval flow, this does **not** set up an optimizer, apply initializers, or attach the model to
        a trainer — it just builds the model, loads the state dict, moves it to ``device``, and puts it in eval mode.

        Args:
            checkpoint: Checkpoint tag (run-dir mode only). Defaults to ``"latest"``. Other examples: ``"E10"``,
                ``"best_model.loss.test.total"``. Ignored in checkpoint-only mode — the file was already fixed at
                :meth:`from_checkpoint` time.
            device: Torch device (or string) to move the model to.

        Returns:
            The model in eval mode with weights loaded.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist (run-dir mode).
            KeyError: If the checkpoint is missing ``state_dict``.
            RuntimeError: If loading the state dict did not actually change the model weights (sanity check against
                silently missing or mismatched keys).
        """
        if self.is_checkpoint_only:
            if self._checkpoint_data is None or self.checkpoint_path is None:
                raise RuntimeError("Run is in checkpoint-only mode but _checkpoint_data / checkpoint_path is None.")
            ckpt = self._checkpoint_data
            config_cls = resolve_config_class(ckpt[CheckpointKeys.CONFIG_KIND], ModelBaseConfig)
            model_config = config_cls.model_validate(ckpt[CheckpointKeys.MODEL_CONFIG])
            model: nn.Module = Factory().instantiate(model_config)
            state_dict = ckpt[CheckpointKeys.STATE_DICT]
            source: Path = self.checkpoint_path
        else:
            if self._config is None:
                raise RuntimeError("Run._config is None in run-dir mode — Run.config may have been cleared.")
            model = Factory().instantiate(self._config.model)
            model_name: str = model.name  # type: ignore[attr-defined,assignment]
            source = self._resolve_checkpoint_path(model_name, checkpoint)
            ckpt = torch.load(source, map_location=device, weights_only=True)
            if CheckpointKeys.STATE_DICT not in ckpt:
                raise KeyError(f"state_dict not found in checkpoint {source}")
            state_dict = ckpt[CheckpointKeys.STATE_DICT]

        norm_before = compute_model_norm(model).item()
        model.load_state_dict(state_dict)
        if compute_model_norm(model).item() == norm_before:
            raise RuntimeError(
                f"model weights unchanged after loading {source} — "
                "the checkpoint may be empty or the state-dict keys may not match the model."
            )

        model.to(device)
        model.eval()
        return model

    def _resolve_checkpoint_path(self, model_name: str, checkpoint: str) -> Path:
        """Resolve ``{run_dir}/checkpoints/{model_name}_cp={checkpoint}_model.th``."""
        if self.run_dir is None:
            raise RuntimeError("Run._resolve_checkpoint_path called in checkpoint-only mode (run_dir is None).")
        ckpt_path = self.run_dir / "checkpoints" / f"{model_name}_cp={checkpoint}_model.th"
        if not ckpt_path.exists():
            available = (
                sorted(p.name for p in (self.run_dir / "checkpoints").glob("*_model.th"))
                if (self.run_dir / "checkpoints").exists()
                else []
            )
            raise FileNotFoundError(
                f"checkpoint not found: {ckpt_path}. "
                f"Available model checkpoints in {self.run_dir / 'checkpoints'}: {available}"
            )
        return ckpt_path
