#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Tests that ``TrainingDiagnosticsCallback`` is configurable from YAML/config.

The callback used to be auto-added in code only and had no resolvable config class, so a
``callbacks:`` entry referencing it failed schema validation. It is now decorated with
``@ConfiguredBy(CallBackBaseConfig)`` so it can be enabled via config (used by the
ShapeNetCar + AB-UPT end-to-end test to track convergence / training dynamics).
"""

from __future__ import annotations

import pytest

from noether.core.callbacks.base import CallBackBaseConfig
from noether.core.callbacks.default.training_diagnostics import TrainingDiagnosticsCallback
from noether.core.providers.metric_property import MetricPropertyProvider, Ordinality
from noether.core.schemas.lib import _discriminated_validator, resolve_config_class

_KIND = "noether.core.callbacks.default.training_diagnostics.TrainingDiagnosticsCallback"


def test_resolves_to_callback_base_config():
    assert resolve_config_class(_KIND, CallBackBaseConfig) is CallBackBaseConfig


def test_config_dict_validates_via_discriminated_validator():
    result = _discriminated_validator({"kind": _KIND, "every_n_updates": 50}, registry_cls=CallBackBaseConfig)
    assert isinstance(result, CallBackBaseConfig)
    assert result.every_n_updates == 50


def test_configured_by_attribute_is_set():
    assert TrainingDiagnosticsCallback._config_class is CallBackBaseConfig


@pytest.fixture
def fresh_metric_patterns():
    """Reset the global ``MetricPropertyProvider`` pattern list around a test and restore it after,
    so registering the callback's patterns does not leak into other tests."""
    saved = list(MetricPropertyProvider._PATTERNS)
    MetricPropertyProvider._PATTERNS = []
    try:
        yield
    finally:
        MetricPropertyProvider._PATTERNS = saved


def test_optim_metrics_register_as_neutral(fresh_metric_patterns):
    """The optim diagnostics metrics must resolve to NEUTRAL (not the warning-emitting default of
    higher_is_better=True), and registration must coexist with the provider defaults."""
    TrainingDiagnosticsCallback._register_metric_patterns()

    for key in (
        "training_diagnostics/optim/grad_scaler_scale",
        "training_diagnostics/optim/grad_norm/ab_upt",
        "training_diagnostics/optim/model_norm",
    ):
        assert MetricPropertyProvider.get_ordinality(key) is Ordinality.NEUTRAL

    # Accumulation-step losses keep lower-is-better via the default ``*loss*`` pattern, proving the
    # provider defaults were registered alongside the callback's pattern.
    assert (
        MetricPropertyProvider.get_ordinality("training_diagnostics/accumulation_step/loss/total")
        is Ordinality.LOWER_IS_BETTER
    )


def test_register_metric_patterns_is_idempotent(fresh_metric_patterns):
    """Registering the patterns twice (e.g. one callback instance per stage) must not accumulate
    duplicate entries in the global pattern list."""
    TrainingDiagnosticsCallback._register_metric_patterns()
    after_first = list(MetricPropertyProvider._PATTERNS)
    TrainingDiagnosticsCallback._register_metric_patterns()

    assert MetricPropertyProvider._PATTERNS == after_first
