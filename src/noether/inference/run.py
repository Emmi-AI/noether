#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Notebook-friendly Python API for loading a trained run.

The non-Hydra counterpart to ``noether-eval``: instead of spinning up an
:class:`~noether.inference.runners.InferenceRunner` (with trainer context,
callbacks, tracker, etc.), it gives you a single handle to the run from
which you can pull the resolved config, an instantiated dataset, and a
model with checkpoint weights loaded.

.. code-block:: python

    from noether.inference import Run

    run = Run("/outputs/2026-04-09_abc12")
    # optionally patch config in place — e.g. fix dataset paths for this machine
    for ds in run.config.datasets.values():
        ds.root = "/local/path/to/data"

    dataset = run.dataset("test")
    model = run.model(checkpoint="latest", device="cuda")

For reproducible eval with metrics, callbacks, and full logging, use
``noether-eval`` instead.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

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

__all__ = ["Run", "load_model_from_checkpoint", "load_normalizers_from_checkpoint", "sanitize_hp_resolved"]


def _to_plain_python(obj: Any) -> Any:
    """Recursively convert tuples/sets to lists so the YAML round-trips through ``yaml.safe_dump``."""
    if isinstance(obj, dict):
        return {k: _to_plain_python(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, set, frozenset)):
        return [_to_plain_python(v) for v in obj]
    if isinstance(obj, list):
        return [_to_plain_python(v) for v in obj]
    return obj


def sanitize_hp_resolved(hp_resolved_path: Path) -> Path:
    """Write a tag-free copy of ``hp_resolved.yaml`` to a temp file.

    ``hp_resolved.yaml`` is written with :func:`yaml.dump`, which emits
    ``!!python/tuple`` tags for tuple values (notably ``dataset_statistics``).
    Hydra and pydantic both prefer plain YAML, so we strip the tags by
    re-serializing through :func:`yaml.safe_dump` after coercing tuples to
    lists.
    """
    with open(hp_resolved_path) as f:
        config = yaml.full_load(f)

    tmp_dir = Path(tempfile.mkdtemp(prefix="noether_eval_"))
    safe_path = tmp_dir / "hp_resolved.yaml"
    with open(safe_path, "w") as f:
        yaml.safe_dump(_to_plain_python(config), f, sort_keys=False)
    return safe_path


class Run:
    """Handle to a trained run directory.

    Construction is cheap — it reads ``hp_resolved.yaml`` and validates it
    against :class:`ConfigSchema`, nothing more. The three lazy methods below
    are independent — pick whichever you need:

    - :meth:`model` — instantiate the trained model with weights loaded. Only
      needs the run's config and checkpoint files; works on **any** input dict
      you can construct, not just samples from the original training set.
    - :meth:`normalizers` — build the field normalizers (e.g. for converting
      model predictions back to physical units). Reads only the dataset class's
      ``STATS_FILE``; the data files themselves are not required.
    - :meth:`dataset` — instantiate the dataset. This **does** require the
      original data files to exist at ``config.datasets[split].root``. Use it
      only if you want to iterate the same data the run was trained on.

    Mutate :attr:`config` between construction and the lazy methods to override
    training-time values (typically dataset roots when the run was produced on
    a different machine).

    Args:
        run_dir: Path to the training run output directory (the one that
            contains ``hp_resolved.yaml`` and a ``checkpoints/`` subdirectory).
            Typically ``output_path/run_id`` or ``output_path/run_id/stage_name``.

    Attributes:
        run_dir: Resolved absolute path to the run directory.
        config: Validated :class:`ConfigSchema` loaded from the run's
            ``hp_resolved.yaml``. Safe to mutate before calling the lazy
            methods.

    Raises:
        FileNotFoundError: If ``run_dir`` does not exist or doesn't contain
            ``hp_resolved.yaml``.

    Example:

        .. testcode::
            :skipif: True  # requires a real run directory

            from noether.inference import Run

            # Bring-your-own-data flow: apply the trained model to a custom
            # input dict, then denormalize the predictions.
            run = Run("/outputs/2026-04-09_abc12")
            model = run.model(device="cuda")
            norms = run.normalizers()
            with torch.inference_mode():
                pred = model(**my_inputs)
            pred_phys = norms["surface_pressure"].inverse(pred["surface_pressure"])
    """

    def __init__(self, run_dir: Path | str):
        self.run_dir: Path = Path(run_dir).expanduser().resolve()
        if not self.run_dir.exists():
            raise FileNotFoundError(f"run_dir does not exist: {self.run_dir}")
        self.config: ConfigSchema = self._load_config()

    def _load_config(self) -> ConfigSchema:
        hp_path = self.run_dir / "hp_resolved.yaml"
        if not hp_path.exists():
            raise FileNotFoundError(
                f"hp_resolved.yaml not found in {self.run_dir}. "
                "Make sure run_dir points at a training run output directory "
                "(typically output_path/run_id[/stage_name])."
            )
        with open(sanitize_hp_resolved(hp_path)) as f:
            data = yaml.safe_load(f)

        # ConfigSchema's _resolve_slurm_defaults validator does
        # ``validate_path(output_path, mkdir=True)`` on whatever path the
        # training run wrote — typically a server path that doesn't make
        # sense on this machine. Anchor the loaded config to the local
        # run_dir so the validator's mkdir is a no-op.
        data["output_path"] = str(self.run_dir)
        return ConfigSchema(**data)

    @property
    def statistics(self) -> dict[str, list[float | int]]:
        """Training-time dataset statistics (``config.dataset_statistics`` or ``{}``).

        Convenience accessor for the stat values the training run computed —
        typically per-field means/stds used by the trainer's pipeline. Returns
        an empty dict if the run didn't compute any stats.

        Note: this is separate from the dataset class's static ``STATS_FILE``,
        which :meth:`normalizers` reads.
        """
        return dict(self.config.dataset_statistics or {})

    def normalizers(self, split: str = "test") -> dict[str, ComposePreProcess]:
        """Build the trained run's field normalizers without instantiating its dataset.

        Mirrors what ``dataset.normalizers`` would expose, but constructed
        without requiring the dataset's data files at ``config.datasets[split].root``.
        Useful when you want to apply the trained model to data that isn't
        part of a noether :class:`Dataset` — use
        ``normalizers[field].inverse(prediction)`` to convert model outputs
        back to physical units, or ``normalizers[field](raw_value)`` to
        normalize your own inputs before feeding the model.

        The dataset class itself is still imported (looked up from
        ``config.datasets[split].kind``) to read its ``STATS_FILE`` class
        attribute — the data root, however, is never touched.

        Args:
            split: Dataset key to source the normalizer configs from. Splits
                typically share normalizers; the arg is provided for parity
                with :meth:`dataset`.

        Returns:
            Dict mapping field name (e.g. ``"surface_pressure"``) to a
            :class:`ComposePreProcess`. Empty dict if the config defines no
            normalizers for this split.

        Raises:
            KeyError: If ``split`` is not in ``self.config.datasets``.
        """
        if split not in self.config.datasets:
            raise KeyError(
                f"split {split!r} not found in config.datasets. Available splits: {sorted(self.config.datasets.keys())}"
            )
        dataset_config = self.config.datasets[split]
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

        normalizers: dict[str, ComposePreProcess] = {}
        for key, configs in dataset_config.dataset_normalizers.items():
            configs_list = configs if isinstance(configs, list) else [configs]
            preprocessors = [
                Factory().instantiate(cfg, normalization_key=key, statistics=statistics) for cfg in configs_list
            ]
            normalizers[key] = ComposePreProcess(normalization_key=key, preprocessors=preprocessors)
        return normalizers

    def dataset(self, split: str = "test") -> Dataset:
        """Instantiate the dataset for ``split``.

        Wires up the collator (``dataset.pipeline``) the same way the trainer
        does, so the dataset can be plugged into a
        :class:`torch.utils.data.DataLoader` for batched forward passes.

        Args:
            split: Dataset key (e.g. ``"train"``, ``"val"``, ``"test"``).

        Raises:
            KeyError: If ``split`` is not in ``self.config.datasets``.
        """
        if split not in self.config.datasets:
            raise KeyError(
                f"split {split!r} not found in config.datasets. Available splits: {sorted(self.config.datasets.keys())}"
            )
        dataset_config = self.config.datasets[split]
        dataset: Dataset = DatasetFactory().create(dataset_config)  # type: ignore[assignment]
        pipeline = Factory().create(dataset_config.pipeline)
        if pipeline is not None:
            dataset.pipeline = pipeline
        return dataset

    def model(
        self,
        *,
        checkpoint: str = "latest",
        device: str | torch.device = "cpu",
    ) -> nn.Module:
        """Instantiate the model and load checkpoint weights.

        Unlike the training/eval flow, this does **not** set up an optimizer,
        apply initializers, or attach the model to a trainer — it just builds
        the model, loads the state dict, moves it to ``device``, and puts it
        in eval mode.

        Args:
            checkpoint: Checkpoint tag. Defaults to ``"latest"``. Other
                examples: ``"E10"``, ``"best_model.loss.test.total"``.
            device: Torch device (or string) to move the model to.

        Returns:
            The model in eval mode with weights loaded.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            RuntimeError: If loading the state dict did not actually change the
                model weights (sanity check against silently missing or
                mismatched keys).
        """
        model: nn.Module = Factory().instantiate(self.config.model)
        model_name: str = model.name  # type: ignore[assignment]
        ckpt_uri = self._checkpoint_path(model_name, checkpoint)

        checkpoint_data = torch.load(ckpt_uri, map_location=device, weights_only=True)
        if CheckpointKeys.STATE_DICT not in checkpoint_data:
            raise KeyError(f"state_dict not found in checkpoint {ckpt_uri}")

        norm_before = compute_model_norm(model).item()
        model.load_state_dict(checkpoint_data[CheckpointKeys.STATE_DICT])
        if compute_model_norm(model).item() == norm_before:
            raise RuntimeError(
                f"model weights unchanged after loading {ckpt_uri} — "
                "the checkpoint may be empty or the state-dict keys may not match the model."
            )

        model.to(device)
        model.eval()
        return model

    def _checkpoint_path(self, model_name: str, checkpoint: str) -> Path:
        """Resolve ``{run_dir}/checkpoints/{model_name}_cp={checkpoint}_model.th``."""
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


def load_model_from_checkpoint(
    checkpoint_path: Path | str,
    *,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Instantiate and load a model from a single checkpoint file.

    The lightweight counterpart to :class:`Run` — use it when you only have a
    ``..._model.th`` file in hand and not the full training run directory.
    Every checkpoint written by noether's ``CheckpointWriter`` embeds the
    model config (:attr:`CheckpointKeys.MODEL_CONFIG`) and the discriminator
    kind (:attr:`CheckpointKeys.CONFIG_KIND`) alongside the weights, which is
    enough to reconstruct the model without ``hp_resolved.yaml``.

    The model class itself must still be importable in the current process
    — the kind string points at a class, not at its implementation. If the
    checkpoint references a recipe-specific model, make sure that recipe is
    installed (or on :data:`sys.path`) before calling.

    Args:
        checkpoint_path: Path to a ``..._model.th`` file written by noether.
        device: Torch device (or string) to move the model to.

    Returns:
        The model in eval mode with weights loaded.

    Raises:
        KeyError: If the checkpoint is missing any of ``state_dict``,
            ``model_config``, or ``config_kind`` (older checkpoints predate
            the embedded config — fall back to :class:`Run` against the run
            directory).
        RuntimeError: If loading the state dict did not actually change the
            model weights (same sanity check as :meth:`Run.model`).
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    for required in (CheckpointKeys.STATE_DICT, CheckpointKeys.CONFIG_KIND, CheckpointKeys.MODEL_CONFIG):
        if required not in ckpt:
            raise KeyError(
                f"checkpoint at {checkpoint_path} is missing {required!r}. "
                "Older runs predate the embedded model config — load via Run() "
                "against the run directory instead."
            )

    config_cls = resolve_config_class(ckpt[CheckpointKeys.CONFIG_KIND], ModelBaseConfig)
    model_config = config_cls.model_validate(ckpt[CheckpointKeys.MODEL_CONFIG])
    model: nn.Module = Factory().instantiate(model_config)

    norm_before = compute_model_norm(model).item()
    model.load_state_dict(ckpt[CheckpointKeys.STATE_DICT])
    if compute_model_norm(model).item() == norm_before:
        raise RuntimeError(
            f"model weights unchanged after loading {checkpoint_path} — "
            "the checkpoint may be empty or the state-dict keys may not match the model."
        )

    model.to(device)
    model.eval()
    return model


def load_normalizers_from_checkpoint(checkpoint_path: Path | str) -> dict[str, ComposePreProcess]:
    """Build field normalizers from a single checkpoint file.

    Companion to :func:`load_model_from_checkpoint`. Reads the per-field
    preprocessor configs and resolved statistics that ``CheckpointWriter``
    embeds in every checkpoint (``CheckpointKeys.NORMALIZER_CONFIGS`` /
    ``NORMALIZER_STATISTICS``) and instantiates the same
    :class:`~noether.data.preprocessors.compose.ComposePreProcess` per field
    that :meth:`Run.normalizers` would produce — no run directory, no
    ``hp_resolved.yaml``, no recipe stats file required.

    Args:
        checkpoint_path: Path to a ``..._model.th`` file written by noether.

    Returns:
        Dict mapping field name (e.g. ``"surface_pressure"``) to a
        :class:`ComposePreProcess`. Empty dict if the checkpoint was written
        from a config with no ``dataset_normalizers`` entry.

    Raises:
        KeyError: If the checkpoint predates the embedded normalizer keys.
            Re-train with the current code or fall back to :class:`Run` against
            the run directory.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if CheckpointKeys.NORMALIZER_CONFIGS not in ckpt:
        raise KeyError(
            f"checkpoint at {checkpoint_path} is missing {CheckpointKeys.NORMALIZER_CONFIGS!r}. "
            "Older runs predate embedded normalizer info — re-train with the current code, "
            "or load normalizers via Run(run_dir).normalizers(split) instead."
        )

    configs_dump = ckpt[CheckpointKeys.NORMALIZER_CONFIGS]
    statistics = ckpt.get(CheckpointKeys.NORMALIZER_STATISTICS)

    normalizers: dict[str, ComposePreProcess] = {}
    for key, configs in configs_dump.items():
        configs_list = configs if isinstance(configs, list) else [configs]
        # Each entry in the checkpoint is a plain dict (from `model_dump`) — re-validate
        # back into a pydantic NormalizerConfig so Factory can read its `kind` field.
        validated = [resolve_config_class(c["kind"], NormalizerConfig).model_validate(c) for c in configs_list]
        preprocessors = [Factory().instantiate(cfg, normalization_key=key, statistics=statistics) for cfg in validated]
        normalizers[key] = ComposePreProcess(normalization_key=key, preprocessors=preprocessors)
    return normalizers
