#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import pytest

from noether.core.presets.base import DomainPreset
from noether.core.schemas.models.base import ModelBaseConfig
from noether.core.schemas.optimizers import (
    AdamOptimizerConfig,
    AnyOptimizerConfig,
    MuonOptimizerConfig,
    SGDOptimizerConfig,
)


class _StubPreset(DomainPreset):
    """Minimal concrete ``DomainPreset`` for exercising base-class helpers."""

    @property
    def data_specs(self):
        return None

    @property
    def normalizer_spec(self):
        return {}

    @property
    def excluded_properties(self):
        return None

    def target_properties(self):
        return []

    def build_dataset(self, **_):
        raise NotImplementedError


class TestBuildOptimizerDispatchesToDiscriminatedSubclass:
    @pytest.mark.parametrize(
        ("kind", "expected_cls"),
        [
            ("noether.core.optimizer.Lion", AdamOptimizerConfig),
            ("torch.optim.AdamW", AdamOptimizerConfig),
            ("torch.optim.SGD", SGDOptimizerConfig),
            ("noether.core.optimizer.MuonComposite", MuonOptimizerConfig),
        ],
    )
    def test_returns_correct_subclass_for_kind(self, kind: str, expected_cls: type) -> None:
        cfg = _StubPreset().build_optimizer(kind=kind)
        assert type(cfg) is expected_cls
        assert cfg.kind == kind

    def test_default_kind_is_lion_and_returns_adam_config(self) -> None:
        cfg = _StubPreset().build_optimizer()
        assert isinstance(cfg, AdamOptimizerConfig)
        assert cfg.kind == "noether.core.optimizer.Lion"

    def test_propagates_lr_weight_decay_and_clip(self) -> None:
        cfg = _StubPreset().build_optimizer(lr=3e-4, weight_decay=0.1, clip_grad_norm=2.0)
        assert cfg.lr == pytest.approx(3e-4)
        assert cfg.weight_decay == pytest.approx(0.1)
        assert cfg.clip_grad_norm == pytest.approx(2.0)

    def test_schedule_is_attached_when_end_lr_provided(self) -> None:
        cfg = _StubPreset().build_optimizer(lr=1e-3, end_lr=1e-6)
        assert cfg.schedule_config is not None
        assert cfg.schedule_config.max_value == pytest.approx(1e-3)
        assert cfg.schedule_config.end_value == pytest.approx(1e-6)

    def test_schedule_is_omitted_when_end_lr_is_none(self) -> None:
        cfg = _StubPreset().build_optimizer(end_lr=None)
        assert cfg.schedule_config is None

    def test_result_validates_as_model_optimizer_config(self) -> None:
        """Regression test for the integration that originally broke: feeding the
        ``build_optimizer`` result into ``ModelBaseConfig`` must satisfy the
        ``Field(discriminator="kind")`` constraint."""
        opt = _StubPreset().build_optimizer()
        model_cfg = ModelBaseConfig.model_validate(
            {
                "kind": "single",
                "name": "m",
                "optimizer_config": opt.model_dump(),
            }
        )
        assert isinstance(model_cfg.optimizer_config, AdamOptimizerConfig)
        # Sanity: the type matches one of the union members.
        assert isinstance(model_cfg.optimizer_config, AnyOptimizerConfig.__args__)  # type: ignore[attr-defined]
