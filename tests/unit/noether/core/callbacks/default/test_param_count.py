#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from noether.core.callbacks.default.param_count import ParamCountCallback
from noether.core.models import CompositeModel
from noether.core.utils.common import snake_type_name


class SimpleModel:
    """Minimal stand-in for a non-composite model."""

    def __init__(self, trainable: int, frozen: int):
        self.trainable_param_count = trainable
        self.frozen_param_count = frozen


# Pre-compute the name that snake_type_name produces for SimpleModel.
_SIMPLE_MODEL_NAME = snake_type_name(SimpleModel(0, 0))


def _make_composite_model(submodels: dict):
    """Create a CompositeModel mock that passes isinstance checks."""
    m = Mock(spec=CompositeModel)
    m.submodels = submodels
    return m


def _make_callback(model) -> ParamCountCallback:
    return ParamCountCallback(
        trainer=SimpleNamespace(),
        model=model,
        data_container=SimpleNamespace(),
        tracker=Mock(),
        log_writer=SimpleNamespace(),
        checkpoint_writer=SimpleNamespace(),
        metric_property_provider=SimpleNamespace(),
        name=None,
    )


class TestGetParamCounts:
    def test_simple_model_no_trace(self):
        model = SimpleModel(trainable=1000, frozen=500)
        result = ParamCountCallback._get_param_counts(model)

        assert len(result) == 2
        # First entry is the total (name=None)
        assert result[0] == (None, 1000, 500)
        # Second entry is the model-specific entry
        assert result[1] == (_SIMPLE_MODEL_NAME, 1000, 500)

    def test_simple_model_with_trace(self):
        """With a trace (non-top-level), no extra total entry is added."""
        model = SimpleModel(trainable=200, frozen=100)
        result = ParamCountCallback._get_param_counts(model, trace="root")

        assert len(result) == 1
        assert result[0] == (f"root.{_SIMPLE_MODEL_NAME}", 200, 100)

    def test_composite_model_with_two_submodels(self):
        sub_a = SimpleModel(trainable=1000, frozen=200)
        sub_b = SimpleModel(trainable=3000, frozen=800)
        model = _make_composite_model({"encoder": sub_a, "decoder": sub_b})

        result = ParamCountCallback._get_param_counts(model, trace="model")

        # First entry is the composite summary, followed by each submodel
        assert len(result) == 3
        # Composite summary: sums of immediate children
        assert result[0] == ("model", 4000, 1000)
        # Submodels (order follows dict iteration order)
        assert result[1] == (f"model.encoder.{_SIMPLE_MODEL_NAME}", 1000, 200)
        assert result[2] == (f"model.decoder.{_SIMPLE_MODEL_NAME}", 3000, 800)

    def test_composite_model_skips_none_submodels(self):
        sub_a = SimpleModel(trainable=500, frozen=100)
        model = _make_composite_model({"encoder": sub_a, "decoder": None})

        result = ParamCountCallback._get_param_counts(model, trace="model")

        assert len(result) == 2
        assert result[0] == ("model", 500, 100)
        assert result[1] == (f"model.encoder.{_SIMPLE_MODEL_NAME}", 500, 100)

    def test_nested_composite_model(self):
        leaf_a = SimpleModel(trainable=100, frozen=10)
        leaf_b = SimpleModel(trainable=200, frozen=20)
        inner_composite = _make_composite_model({"a": leaf_a, "b": leaf_b})
        leaf_c = SimpleModel(trainable=300, frozen=30)
        outer_composite = _make_composite_model({"inner": inner_composite, "head": leaf_c})

        result = ParamCountCallback._get_param_counts(outer_composite, trace="model")

        # outer summary + inner summary + leaf_a + leaf_b + leaf_c = 5 entries
        assert len(result) == 5
        # Outer summary: 100+200+300=600 trainable, 10+20+30=60 frozen
        assert result[0] == ("model", 600, 60)
        # Inner composite summary
        assert result[1] == ("model.inner", 300, 30)
        # Leaves
        assert result[2] == (f"model.inner.a.{_SIMPLE_MODEL_NAME}", 100, 10)
        assert result[3] == (f"model.inner.b.{_SIMPLE_MODEL_NAME}", 200, 20)
        assert result[4] == (f"model.head.{_SIMPLE_MODEL_NAME}", 300, 30)

    def test_composite_model_all_none_submodels(self):
        model = _make_composite_model({"a": None, "b": None})

        result = ParamCountCallback._get_param_counts(model, trace="model")

        # Only the composite summary with zero counts
        assert len(result) == 1
        assert result[0] == ("model", 0, 0)


class TestBeforeTraining:
    def test_logs_and_updates_tracker_for_simple_model(self):
        model = SimpleModel(trainable=1_500_000, frozen=500_000)
        cb = _make_callback(model)

        cb.before_training()

        cb.tracker.update_summary.assert_called_once()
        summary = cb.tracker.update_summary.call_args[0][0]
        # Total entry
        assert summary["param_count/total/trainable"] == 1_500_000
        assert summary["param_count/total/frozen"] == 500_000
        assert summary["param_count/total"] == 2_000_000
        # Model-specific entry
        assert summary[f"param_count/{_SIMPLE_MODEL_NAME}/trainable"] == 1_500_000
        assert summary[f"param_count/{_SIMPLE_MODEL_NAME}/frozen"] == 500_000
        assert summary[f"param_count/{_SIMPLE_MODEL_NAME}"] == 2_000_000

    def test_logs_and_updates_tracker_for_composite_model(self):
        sub_a = SimpleModel(trainable=1000, frozen=200)
        sub_b = SimpleModel(trainable=3000, frozen=800)
        model = _make_composite_model({"encoder": sub_a, "decoder": sub_b})
        cb = _make_callback(model)

        cb.before_training()

        summary = cb.tracker.update_summary.call_args[0][0]
        # The composite root has trace=None, so name becomes "total"
        assert summary["param_count/total/trainable"] == 4000
        assert summary["param_count/total/frozen"] == 1000
        assert summary["param_count/total"] == 5000
        # Submodel entries
        assert f"param_count/None.encoder.{_SIMPLE_MODEL_NAME}/trainable" in summary
        assert f"param_count/None.decoder.{_SIMPLE_MODEL_NAME}/trainable" in summary

    @pytest.mark.parametrize(
        "model",
        [
            SimpleModel(trainable=100, frozen=0),
            _make_composite_model({"part": SimpleModel(trainable=100, frozen=0)}),
        ],
        ids=["simple", "composite"],
    )
    def test_always_outputs_total(self, model):
        """Both simple and composite models produce a 'total' entry."""
        cb = _make_callback(model)

        cb.before_training()

        summary = cb.tracker.update_summary.call_args[0][0]
        assert "param_count/total/trainable" in summary
        assert "param_count/total/frozen" in summary
        assert "param_count/total" in summary

    def test_logs_info_messages(self, caplog):
        model = SimpleModel(trainable=1000, frozen=500)
        cb = _make_callback(model)

        with caplog.at_level("INFO"):
            cb.before_training()

        assert any("Parameter count" in record.message for record in caplog.records)
        assert any("trainable=" in record.message for record in caplog.records)
        assert any("frozen=" in record.message for record in caplog.records)
