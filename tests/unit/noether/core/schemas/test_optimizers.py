#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
from pydantic import TypeAdapter, ValidationError

from noether.core.models.base import ModelBaseConfig
from noether.core.optimizer.schemas import (
    AdamOptimizerConfig,
    AnyOptimizerConfig,
    MuonOptimizerConfig,
    SGDOptimizerConfig,
)

_adapter = TypeAdapter(AnyOptimizerConfig)


class TestOptimizerConfigDispatch:
    @pytest.mark.parametrize(
        ("kind", "expected_cls"),
        [
            ("torch.optim.AdamW", AdamOptimizerConfig),
            ("noether.core.optimizer.Lion", AdamOptimizerConfig),
            ("torch.optim.SGD", SGDOptimizerConfig),
            ("noether.core.optimizer.MuonComposite", MuonOptimizerConfig),
        ],
    )
    def test_known_kinds_dispatch_to_subclass(self, kind, expected_cls):
        cfg = _adapter.validate_python({"kind": kind, "lr": 1e-3})
        assert type(cfg) is expected_cls

    def test_unknown_kind_is_rejected(self):
        with pytest.raises(ValidationError):
            _adapter.validate_python({"kind": "torch.optim.RMSprop", "lr": 1e-3})

    @pytest.mark.parametrize(
        "bad_config",
        [
            {"kind": "torch.optim.AdamW", "lr": 1e-3, "momentum": 0.9},
            {"kind": "torch.optim.AdamW", "lr": 1e-3, "secondary": {}},
            {"kind": "torch.optim.SGD", "lr": 1e-2, "betas": (0.9, 0.99)},
            {"kind": "torch.optim.SGD", "lr": 1e-2, "secondary": {}},
            {"kind": "noether.core.optimizer.MuonComposite", "lr": 2e-3, "betas": (0.9, 0.99)},
        ],
    )
    def test_invalid_field_combinations_rejected(self, bad_config):
        with pytest.raises(ValidationError):
            _adapter.validate_python(bad_config)


class TestModelBaseConfigDispatch:
    """The production use path nests the Union under a Field(discriminator="kind")."""

    def test_nested_discriminator_dispatches(self):
        model_cfg = ModelBaseConfig.model_validate(
            {
                "kind": "single",
                "name": "m",
                "optimizer_config": {"kind": "torch.optim.SGD", "lr": 0.1, "momentum": 0.9},
            }
        )
        assert isinstance(model_cfg.optimizer_config, SGDOptimizerConfig)
        assert model_cfg.optimizer_config.momentum == 0.9
