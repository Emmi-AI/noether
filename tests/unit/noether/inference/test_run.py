#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import yaml
from torch import nn

from noether.core.schemas.schema import ConfigSchema
from noether.core.types import CheckpointKeys
from noether.inference.run import Run, sanitize_hp_resolved

_MODULE_PATH = "noether.inference.run"


# ---------------------------------------------------------------------------
# sanitize_hp_resolved
# ---------------------------------------------------------------------------


class TestSanitizeHpResolved:
    """The sanitize helper strips ``!!python/...`` tags so pydantic / Hydra can read the file."""

    def test_strips_python_tuple_tags(self, tmp_path):
        original = tmp_path / "hp_resolved.yaml"
        original.write_text(yaml.dump({"shape": (1, 2, 3), "name": "abc"}))
        assert "!!python/tuple" in original.read_text()

        safe = sanitize_hp_resolved(original)

        sanitized = safe.read_text()
        assert "!!python/tuple" not in sanitized
        assert yaml.safe_load(sanitized) == {"shape": [1, 2, 3], "name": "abc"}

    def test_returns_path_in_fresh_tempdir(self, tmp_path):
        original = tmp_path / "hp_resolved.yaml"
        original.write_text(yaml.dump({"name": "abc"}))

        safe = sanitize_hp_resolved(original)

        assert safe.name == "hp_resolved.yaml"
        assert safe.parent != original.parent


# ---------------------------------------------------------------------------
# Run.__init__ / config loading
# ---------------------------------------------------------------------------


def _make_run_dir(tmp_path: Path, config_data: dict) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "hp_resolved.yaml").write_text(yaml.dump(config_data))
    return run_dir


class TestRunInit:
    """``Run(run_dir)`` reads + validates hp_resolved.yaml; no heavy work."""

    def test_dispatches_to_config_schema_with_sanitized_yaml(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, {"shape": (1, 2, 3), "name": "abc"})  # tuple -> !!python/tuple

        sentinel = MagicMock(spec=ConfigSchema)
        with patch(_MODULE_PATH + ".ConfigSchema", return_value=sentinel) as mock_cls:
            run = Run(run_dir)

        assert run.config is sentinel
        assert run.run_dir == run_dir.resolve()
        (call_kwargs,) = [c.kwargs for c in mock_cls.call_args_list]
        assert call_kwargs["shape"] == [1, 2, 3]
        assert call_kwargs["name"] == "abc"

    def test_overrides_output_path_to_run_dir_to_avoid_mkdir_side_effect(self, tmp_path):
        """ConfigSchema's validator does ``validate_path(output_path, mkdir=True)``.
        We point ``output_path`` at the local run_dir so the validator's mkdir is a no-op,
        rather than silently creating whatever path the training run wrote to."""
        run_dir = _make_run_dir(tmp_path, {"output_path": "/some/path/that/should/not/be/created"})

        with patch(_MODULE_PATH + ".ConfigSchema") as mock_cls:
            Run(run_dir)

        (call_kwargs,) = [c.kwargs for c in mock_cls.call_args_list]
        assert call_kwargs["output_path"] == str(run_dir.resolve())
        # Sanity: the bogus path from the source config wasn't created.
        assert not Path("/some/path/that/should/not/be/created").exists()

    def test_accepts_string_path(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, {"name": "abc"})

        with patch(_MODULE_PATH + ".ConfigSchema"):
            run = Run(str(run_dir))

        assert run.run_dir == run_dir.resolve()

    def test_raises_when_run_dir_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="run_dir does not exist"):
            Run(tmp_path / "nope")

    def test_raises_when_hp_resolved_missing(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="hp_resolved.yaml not found"):
            Run(run_dir)


# ---------------------------------------------------------------------------
# Run.statistics / Run.normalizers (BYO-data accessors)
# ---------------------------------------------------------------------------


def _make_run_with_mocked_config_static(tmp_path: Path) -> Run:
    """Build a Run whose ``config`` is a freshly mocked instance — caller wires fields."""
    run_dir = _make_run_dir(tmp_path, {"name": "abc"})
    with patch(_MODULE_PATH + ".ConfigSchema"):
        run = Run(run_dir)
    run.config = MagicMock(spec=ConfigSchema)
    return run


class TestRunStatistics:
    def test_returns_config_dataset_statistics(self, tmp_path):
        run = _make_run_with_mocked_config_static(tmp_path)
        run.config.dataset_statistics = {"surface_pressure_mean": [10.0]}
        assert run.statistics == {"surface_pressure_mean": [10.0]}

    def test_returns_empty_dict_when_none(self, tmp_path):
        run = _make_run_with_mocked_config_static(tmp_path)
        run.config.dataset_statistics = None
        assert run.statistics == {}


class _FakeDatasetWithStatsFile:
    """Stand-in for a noether Dataset class whose ``STATS_FILE`` attribute points at a real YAML."""

    STATS_FILE: str  # set per-test


class TestRunNormalizers:
    """``Run.normalizers()`` builds normalizers without instantiating the dataset class."""

    def test_returns_empty_dict_when_no_normalizers_in_config(self, tmp_path):
        run = _make_run_with_mocked_config_static(tmp_path)
        ds_cfg = MagicMock()
        ds_cfg.dataset_normalizers = None
        run.config.datasets = {"test": ds_cfg}

        assert run.normalizers("test") == {}

    def test_raises_on_unknown_split(self, tmp_path):
        run = _make_run_with_mocked_config_static(tmp_path)
        run.config.datasets = {"train": MagicMock()}
        with pytest.raises(KeyError, match=r"'test'"):
            run.normalizers("test")

    def test_builds_normalizers_with_stats_from_dataset_class_stats_file(self, tmp_path):
        """Loads the dataset class via the ``kind`` field, reads its ``STATS_FILE``,
        and passes the resulting stats dict to each normalizer's constructor —
        no dataset instantiation, no data root access."""
        stats_path = tmp_path / "stats.yaml"
        stats_path.write_text(yaml.safe_dump({"surface_pressure_mean": [10.0], "surface_pressure_std": 2.0}))

        _FakeDatasetWithStatsFile.STATS_FILE = str(stats_path)

        run = _make_run_with_mocked_config_static(tmp_path)
        normalizer_cfg = MagicMock()
        ds_cfg = MagicMock()
        ds_cfg.kind = "some.fake.dataset.kind"
        ds_cfg.dataset_normalizers = {"surface_pressure": normalizer_cfg}
        run.config.datasets = {"test": ds_cfg}

        with (
            patch(_MODULE_PATH + ".class_constructor_from_class_path", return_value=_FakeDatasetWithStatsFile),
            patch(_MODULE_PATH + ".Factory") as mock_factory_cls,
            patch(_MODULE_PATH + ".ComposePreProcess") as mock_compose,
        ):
            mock_compose.side_effect = lambda **kw: MagicMock(key=kw["normalization_key"])
            normalizers = run.normalizers("test")

        assert "surface_pressure" in normalizers
        # Stats were passed through as a coerced dict; check the call.
        (call_kwargs,) = [c.kwargs for c in mock_factory_cls.return_value.instantiate.call_args_list]
        assert call_kwargs["normalization_key"] == "surface_pressure"
        assert call_kwargs["statistics"] == {"surface_pressure_mean": [10.0], "surface_pressure_std": 2.0}

    def test_handles_dataset_class_without_stats_file(self, tmp_path):
        """If the dataset class has no ``STATS_FILE``, statistics is None — the
        normalizer constructor decides whether it can cope (e.g. some normalizers
        don't need stats)."""

        class _NoStatsDataset:
            pass

        run = _make_run_with_mocked_config_static(tmp_path)
        normalizer_cfg = MagicMock()
        ds_cfg = MagicMock()
        ds_cfg.kind = "some.fake.dataset.kind"
        ds_cfg.dataset_normalizers = {"f": normalizer_cfg}
        run.config.datasets = {"test": ds_cfg}

        with (
            patch(_MODULE_PATH + ".class_constructor_from_class_path", return_value=_NoStatsDataset),
            patch(_MODULE_PATH + ".Factory") as mock_factory_cls,
            patch(_MODULE_PATH + ".ComposePreProcess"),
        ):
            run.normalizers("test")

        (call_kwargs,) = [c.kwargs for c in mock_factory_cls.return_value.instantiate.call_args_list]
        assert call_kwargs["statistics"] is None

    def test_supports_list_of_normalizer_configs_per_field(self, tmp_path):
        """``dataset_normalizers[field]`` can be a single config or a list; both must work."""
        stats_path = tmp_path / "stats.yaml"
        stats_path.write_text(yaml.safe_dump({}))
        _FakeDatasetWithStatsFile.STATS_FILE = str(stats_path)

        run = _make_run_with_mocked_config_static(tmp_path)
        ds_cfg = MagicMock()
        ds_cfg.kind = "some.fake.dataset.kind"
        ds_cfg.dataset_normalizers = {"f": [MagicMock(), MagicMock()]}
        run.config.datasets = {"test": ds_cfg}

        with (
            patch(_MODULE_PATH + ".class_constructor_from_class_path", return_value=_FakeDatasetWithStatsFile),
            patch(_MODULE_PATH + ".Factory") as mock_factory_cls,
            patch(_MODULE_PATH + ".ComposePreProcess"),
        ):
            run.normalizers("test")

        # Both configs in the list got instantiated.
        assert mock_factory_cls.return_value.instantiate.call_count == 2


# ---------------------------------------------------------------------------
# Run.dataset
# ---------------------------------------------------------------------------


def _make_run_with_mocked_config(tmp_path: Path, datasets: dict) -> Run:
    """Build a Run whose ``config`` is a mock with the given datasets dict."""
    run_dir = _make_run_dir(tmp_path, {"name": "abc"})
    with patch(_MODULE_PATH + ".ConfigSchema"):
        run = Run(run_dir)
    run.config = MagicMock(spec=ConfigSchema)
    run.config.datasets = datasets
    run.config.model = MagicMock()
    return run


class TestRunDataset:
    def test_dispatches_to_dataset_factory_and_wires_pipeline(self, tmp_path):
        test_cfg = MagicMock()
        test_cfg.pipeline = MagicMock()
        run = _make_run_with_mocked_config(tmp_path, {"test": test_cfg, "train": MagicMock()})

        with (
            patch(_MODULE_PATH + ".DatasetFactory") as mock_ds_factory_cls,
            patch(_MODULE_PATH + ".Factory") as mock_factory_cls,
        ):
            result = run.dataset("test")

        mock_ds_factory_cls.return_value.create.assert_called_once_with(test_cfg)
        # Same wiring as the trainer: collator is built from dataset_config.pipeline and attached.
        mock_factory_cls.return_value.create.assert_called_once_with(test_cfg.pipeline)
        assert result is mock_ds_factory_cls.return_value.create.return_value
        assert result.pipeline is mock_factory_cls.return_value.create.return_value

    def test_skips_pipeline_assignment_when_factory_returns_none(self, tmp_path):
        """If the pipeline config is empty/None, don't overwrite an existing default pipeline."""
        test_cfg = MagicMock()
        test_cfg.pipeline = None
        run = _make_run_with_mocked_config(tmp_path, {"test": test_cfg})

        with (
            patch(_MODULE_PATH + ".DatasetFactory") as mock_ds_factory_cls,
            patch(_MODULE_PATH + ".Factory") as mock_factory_cls,
        ):
            mock_factory_cls.return_value.create.return_value = None
            dataset_returned = mock_ds_factory_cls.return_value.create.return_value
            dataset_returned.pipeline = "untouched"

            result = run.dataset("test")

        assert result.pipeline == "untouched"

    def test_raises_on_unknown_split_listing_available_keys(self, tmp_path):
        run = _make_run_with_mocked_config(tmp_path, {"train": MagicMock(), "val": MagicMock()})

        with pytest.raises(KeyError, match=r"'test'"):
            run.dataset("test")

        with pytest.raises(KeyError, match=r"\['train', 'val'\]"):
            run.dataset("test")

    def test_default_split_is_test(self, tmp_path):
        test_cfg = MagicMock()
        test_cfg.pipeline = None
        run = _make_run_with_mocked_config(tmp_path, {"test": test_cfg})

        with (
            patch(_MODULE_PATH + ".DatasetFactory") as mock_ds_factory_cls,
            patch(_MODULE_PATH + ".Factory"),
        ):
            run.dataset()

        mock_ds_factory_cls.return_value.create.assert_called_once_with(test_cfg)


# ---------------------------------------------------------------------------
# Run.model + checkpoint resolution
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Minimal nn.Module that mimics the parts of ``Model`` that Run.model touches."""

    def __init__(self, name: str = "ab_upt"):
        super().__init__()
        self.name = name
        self.linear = nn.Linear(3, 2)


def _write_checkpoint(path: Path, state_dict: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({CheckpointKeys.STATE_DICT: state_dict}, path)


class TestRunModel:
    """``run.model()`` instantiates via Factory, loads weights, runs sanity check, eval()s."""

    def _setup_run_with_checkpoint(self, tmp_path: Path, state_dict: dict, tag: str = "latest") -> Run:
        run_dir = _make_run_dir(tmp_path, {"name": "abc"})
        _write_checkpoint(run_dir / "checkpoints" / f"ab_upt_cp={tag}_model.th", state_dict)
        with patch(_MODULE_PATH + ".ConfigSchema"):
            run = Run(run_dir)
        run.config = MagicMock(spec=ConfigSchema)
        run.config.model = MagicMock()
        return run

    def test_loads_weights_and_returns_eval_mode_model(self, tmp_path):
        model = _TinyModel()
        target_state = {
            "linear.weight": torch.full_like(model.linear.weight, 0.5),
            "linear.bias": torch.full_like(model.linear.bias, 0.5),
        }
        run = self._setup_run_with_checkpoint(tmp_path, target_state)

        with patch(_MODULE_PATH + ".Factory") as mock_factory_cls:
            mock_factory_cls.return_value.instantiate.return_value = model
            result = run.model()

        assert result is model
        assert not model.training
        assert torch.allclose(model.linear.weight, torch.full_like(model.linear.weight, 0.5))
        assert torch.allclose(model.linear.bias, torch.full_like(model.linear.bias, 0.5))

    def test_raises_when_weights_unchanged(self, tmp_path):
        """Sanity check from ResumeInitializer: identical state_dict => probably wrong key set."""
        model = _TinyModel()
        identical_state = {k: v.clone() for k, v in model.state_dict().items()}
        run = self._setup_run_with_checkpoint(tmp_path, identical_state)

        with patch(_MODULE_PATH + ".Factory") as mock_factory_cls:
            mock_factory_cls.return_value.instantiate.return_value = model
            with pytest.raises(RuntimeError, match="weights unchanged"):
                run.model()

    def test_raises_when_state_dict_key_missing(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, {"name": "abc"})
        ckpt_path = run_dir / "checkpoints" / "ab_upt_cp=latest_model.th"
        ckpt_path.parent.mkdir(parents=True)
        torch.save({"something_else": 1}, ckpt_path)

        with patch(_MODULE_PATH + ".ConfigSchema"):
            run = Run(run_dir)
        run.config = MagicMock(spec=ConfigSchema)
        run.config.model = MagicMock()

        model = _TinyModel()
        with patch(_MODULE_PATH + ".Factory") as mock_factory_cls:
            mock_factory_cls.return_value.instantiate.return_value = model
            with pytest.raises(KeyError, match="state_dict not found"):
                run.model()

    def test_honors_explicit_checkpoint_tag(self, tmp_path):
        model = _TinyModel()
        target_state = {
            "linear.weight": torch.full_like(model.linear.weight, 0.5),
            "linear.bias": torch.full_like(model.linear.bias, 0.5),
        }
        run = self._setup_run_with_checkpoint(tmp_path, target_state, tag="E10")

        with patch(_MODULE_PATH + ".Factory") as mock_factory_cls:
            mock_factory_cls.return_value.instantiate.return_value = model
            run.model(checkpoint="E10")

    def test_checkpoint_missing_lists_available_files(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, {"name": "abc"})
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "ab_upt_cp=E5_model.th").write_bytes(b"")
        (ckpt_dir / "ab_upt_cp=E10_model.th").write_bytes(b"")

        with patch(_MODULE_PATH + ".ConfigSchema"):
            run = Run(run_dir)
        run.config = MagicMock(spec=ConfigSchema)
        run.config.model = MagicMock()

        with patch(_MODULE_PATH + ".Factory") as mock_factory_cls:
            mock_factory_cls.return_value.instantiate.return_value = _TinyModel()
            with pytest.raises(FileNotFoundError, match=r"E5_model.th"):
                run.model()


# ---------------------------------------------------------------------------
# Run.from_checkpoint — single-file flow
# ---------------------------------------------------------------------------


def _write_full_checkpoint(
    path: Path,
    state_dict: dict,
    *,
    include_kind: bool = True,
    include_config: bool = True,
    normalizer_configs: dict | None = None,
    normalizer_statistics: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt: dict = {CheckpointKeys.STATE_DICT: state_dict}
    if include_kind:
        ckpt[CheckpointKeys.CONFIG_KIND] = "fake.model.kind"
    if include_config:
        ckpt[CheckpointKeys.MODEL_CONFIG] = {"some": "config"}
    if normalizer_configs is not None:
        ckpt[CheckpointKeys.NORMALIZER_CONFIGS] = normalizer_configs
    if normalizer_statistics is not None:
        ckpt[CheckpointKeys.NORMALIZER_STATISTICS] = normalizer_statistics
    torch.save(ckpt, path)


def _patch_resolve_model(model: nn.Module):
    """Mock ``resolve_config_class`` + ``Factory`` so checkpoint-mode .model() can run
    without a real ``ModelBaseConfig`` subclass on the registry."""
    resolved_cls = MagicMock()
    resolved_cls.model_validate.return_value = MagicMock()
    rcc_patch = patch(_MODULE_PATH + ".resolve_config_class", return_value=resolved_cls)
    factory_patch = patch(_MODULE_PATH + ".Factory")
    return rcc_patch, factory_patch, resolved_cls


def _patch_resolve_passthrough():
    """Like ``_patch_resolve_model`` but ``model_validate`` returns the dict it got —
    used by the normalizer path, where the dict is fed back into Factory."""
    resolved_cls = MagicMock()
    resolved_cls.model_validate.side_effect = lambda d: d
    return patch(_MODULE_PATH + ".resolve_config_class", return_value=resolved_cls)


class TestRunFromCheckpoint:
    """``Run.from_checkpoint(path)`` — single-file companion to ``Run(run_dir)``.

    The returned Run can produce a model and normalizers from the embedded
    metadata, but ``config`` / ``dataset()`` / ``statistics`` raise because
    there's no resolved schema.
    """

    def test_constructs_in_checkpoint_only_mode(self, tmp_path):
        ckpt_path = tmp_path / "model.th"
        _write_full_checkpoint(ckpt_path, {}, normalizer_configs={"f": {"kind": "fake"}})

        run = Run.from_checkpoint(ckpt_path)

        assert run.run_dir is None
        assert run.checkpoint_path == ckpt_path.resolve()

    def test_raises_when_checkpoint_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="checkpoint not found"):
            Run.from_checkpoint(tmp_path / "nope.th")

    @pytest.mark.parametrize(
        ("missing_kwarg", "missing_key"),
        [
            ({"include_kind": False}, "config_kind"),
            ({"include_config": False}, "model_config"),
        ],
    )
    def test_raises_on_missing_embedded_metadata(self, tmp_path, missing_kwarg, missing_key):
        ckpt_path = tmp_path / "model.th"
        _write_full_checkpoint(ckpt_path, _TinyModel().state_dict(), **missing_kwarg)

        with pytest.raises(KeyError, match=missing_key):
            Run.from_checkpoint(ckpt_path)

    def test_config_and_statistics_raise_in_checkpoint_mode(self, tmp_path):
        """Accessors that depend on the resolved schema fail loudly with a useful message."""
        ckpt_path = tmp_path / "model.th"
        _write_full_checkpoint(ckpt_path, {})
        run = Run.from_checkpoint(ckpt_path)

        with pytest.raises(RuntimeError, match="checkpoint-only"):
            _ = run.config
        with pytest.raises(RuntimeError, match="checkpoint-only"):
            _ = run.statistics

    def test_dataset_raises_in_checkpoint_mode(self, tmp_path):
        ckpt_path = tmp_path / "model.th"
        _write_full_checkpoint(ckpt_path, {})
        run = Run.from_checkpoint(ckpt_path)

        with pytest.raises(RuntimeError, match="checkpoint-only"):
            run.dataset("test")


class TestRunFromCheckpointModel:
    """Model loading via embedded ``MODEL_CONFIG`` + ``CONFIG_KIND``."""

    def test_loads_weights_and_returns_eval_mode_model(self, tmp_path):
        model = _TinyModel()
        target_state = {
            "linear.weight": torch.full_like(model.linear.weight, 0.5),
            "linear.bias": torch.full_like(model.linear.bias, 0.5),
        }
        ckpt_path = tmp_path / "ab_upt_cp=latest_model.th"
        _write_full_checkpoint(ckpt_path, target_state)

        rcc_patch, factory_patch, _ = _patch_resolve_model(model)
        with rcc_patch, factory_patch as factory_mock:
            factory_mock.return_value.instantiate.return_value = model
            run = Run.from_checkpoint(ckpt_path)
            result = run.model()

        assert result is model
        assert not model.training
        assert torch.allclose(model.linear.weight, torch.full_like(model.linear.weight, 0.5))

    def test_uses_kind_and_config_from_checkpoint(self, tmp_path):
        model = _TinyModel()
        target_state = {
            "linear.weight": torch.full_like(model.linear.weight, 0.5),
            "linear.bias": torch.full_like(model.linear.bias, 0.5),
        }
        ckpt_path = tmp_path / "model.th"
        _write_full_checkpoint(ckpt_path, target_state)

        rcc_patch, factory_patch, resolved_cls = _patch_resolve_model(model)
        with rcc_patch as rcc_mock, factory_patch as factory_mock:
            factory_mock.return_value.instantiate.return_value = model
            run = Run.from_checkpoint(ckpt_path)
            run.model()

        # First resolve call is for the model config; checkpoint-mode .model() makes one such call.
        model_calls = [c for c in rcc_mock.call_args_list if c.args[0] == "fake.model.kind"]
        assert len(model_calls) == 1
        resolved_cls.model_validate.assert_called_once_with({"some": "config"})

    def test_raises_when_weights_unchanged(self, tmp_path):
        """Sanity check fires when the loaded state dict matches the freshly-instantiated model."""
        model = _TinyModel()
        ckpt_path = tmp_path / "model.th"
        _write_full_checkpoint(ckpt_path, {k: v.clone() for k, v in model.state_dict().items()})

        rcc_patch, factory_patch, _ = _patch_resolve_model(model)
        with rcc_patch, factory_patch as factory_mock:
            factory_mock.return_value.instantiate.return_value = model
            run = Run.from_checkpoint(ckpt_path)
            with pytest.raises(RuntimeError, match="weights unchanged"):
                run.model()


class TestRunFromCheckpointNormalizers:
    """Normalizer loading via embedded ``NORMALIZER_CONFIGS`` + ``NORMALIZER_STATISTICS``."""

    def test_builds_one_compose_per_field_with_statistics(self, tmp_path):
        ckpt_path = tmp_path / "model.th"
        _write_full_checkpoint(
            ckpt_path,
            {},
            normalizer_configs={"surface_pressure": [{"kind": "fake.Norm"}]},
            normalizer_statistics={"surface_pressure_mean": 1.0, "surface_pressure_std": 2.0},
        )

        sentinel_preprocessor = MagicMock()
        with (
            _patch_resolve_passthrough(),
            patch(_MODULE_PATH + ".Factory") as factory_mock,
            patch(_MODULE_PATH + ".ComposePreProcess") as compose_mock,
        ):
            factory_mock.return_value.instantiate.return_value = sentinel_preprocessor
            compose_mock.return_value = "compose-instance"

            run = Run.from_checkpoint(ckpt_path)
            result = run.normalizers()

        assert result == {"surface_pressure": "compose-instance"}
        # The Factory got the field key + the statistics dict that was in the checkpoint.
        ((kw,),) = [(c.kwargs,) for c in factory_mock.return_value.instantiate.call_args_list]
        assert kw["normalization_key"] == "surface_pressure"
        assert kw["statistics"] == {"surface_pressure_mean": 1.0, "surface_pressure_std": 2.0}
        compose_mock.assert_called_once_with(
            normalization_key="surface_pressure", preprocessors=[sentinel_preprocessor]
        )

    def test_supports_single_config_or_list(self, tmp_path):
        """``NORMALIZER_CONFIGS[field]`` can be a single config dump or a list of them."""
        ckpt_path = tmp_path / "model.th"
        _write_full_checkpoint(
            ckpt_path,
            {},
            normalizer_configs={
                "pressure": {"kind": "fake.Norm"},  # single
                "velocity": [{"kind": "fake.A"}, {"kind": "fake.B"}],  # list
            },
        )

        with (
            _patch_resolve_passthrough(),
            patch(_MODULE_PATH + ".Factory") as factory_mock,
            patch(_MODULE_PATH + ".ComposePreProcess"),
        ):
            factory_mock.return_value.instantiate.return_value = MagicMock()
            run = Run.from_checkpoint(ckpt_path)
            run.normalizers()

        # 1 call for pressure + 2 for velocity
        assert factory_mock.return_value.instantiate.call_count == 3

    def test_raises_when_configs_missing(self, tmp_path):
        ckpt_path = tmp_path / "model.th"
        _write_full_checkpoint(ckpt_path, {})  # no normalizer payload
        run = Run.from_checkpoint(ckpt_path)

        with pytest.raises(KeyError, match="normalizer_configs"):
            run.normalizers()
