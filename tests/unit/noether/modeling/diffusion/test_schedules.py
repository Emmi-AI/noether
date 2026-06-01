#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

"""Unit tests for ``noether.modeling.diffusion`` schedules and config schemas."""

from __future__ import annotations

import pytest
import torch
from pydantic import BaseModel, Field, ValidationError

from noether.modeling.diffusion import (
    AnyDiffusionScheduleConfig,
    DiffusionSchedule,
    FlowMatchingConfig,
    FlowMatchingSchedule,
    build_schedule,
)

# ---------------------------------------------------------------------------
# config / discriminated union
# ---------------------------------------------------------------------------


class _Wrapper(BaseModel):
    schedule: AnyDiffusionScheduleConfig = Field(discriminator="kind")


@pytest.mark.parametrize(
    ("kind", "cfg_cls"),
    [
        ("flow_matching", FlowMatchingConfig),
    ],
)
def test_discriminated_union_resolves_kind(kind, cfg_cls):
    """A dict with the right ``kind`` resolves to the matching config class."""
    obj = _Wrapper.model_validate({"schedule": {"kind": kind}})
    assert isinstance(obj.schedule, cfg_cls)
    assert obj.schedule.kind == kind


def test_discriminated_union_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        _Wrapper.model_validate({"schedule": {"kind": "not_a_paradigm"}})


@pytest.mark.parametrize("cfg_cls", [FlowMatchingConfig])
def test_config_extra_forbid(cfg_cls):
    """Schedule configs reject unknown fields to catch typos in YAML/CLI flags."""
    with pytest.raises(ValidationError):
        cfg_cls(does_not_exist=42)


# ---------------------------------------------------------------------------
# build_schedule factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cfg", "expected"),
    [
        (FlowMatchingConfig(), FlowMatchingSchedule),
    ],
)
def test_build_schedule_dispatches(cfg, expected):
    schedule = build_schedule(cfg)
    assert isinstance(schedule, expected)
    assert isinstance(schedule, DiffusionSchedule)


def test_build_schedule_rejects_unknown():
    class _Other(BaseModel):
        kind: str = "other"

    with pytest.raises(ValueError, match="Unknown diffusion schedule config"):
        build_schedule(_Other())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# schedule.to(device) moves all tensor attributes
# ---------------------------------------------------------------------------


def test_to_returns_self_for_chaining():
    schedule = FlowMatchingSchedule(FlowMatchingConfig())
    assert schedule.to("cpu") is schedule


# ---------------------------------------------------------------------------
# FlowMatching training_losses + sample shape contracts
# ---------------------------------------------------------------------------


def _identity_model(x, t, condition):
    return torch.zeros_like(x)


@pytest.mark.parametrize("continuous_time", [True, False])
def test_fm_training_losses(continuous_time):
    schedule = FlowMatchingSchedule(FlowMatchingConfig(continuous_time=continuous_time)).to("cpu")
    x0 = torch.randn(4, 8)
    loss = schedule.training_losses(x0, _identity_model)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_fm_sample_shape():
    schedule = FlowMatchingSchedule(FlowMatchingConfig()).to("cpu")
    out = schedule.sample(shape=(2, 8), model_fn=_identity_model, steps=4)
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_fm_noise_pair_endpoints():
    """At ``t=1`` ``xt`` should equal ``x1``; at ``t=0`` it should equal the noise."""
    schedule = FlowMatchingSchedule(FlowMatchingConfig()).to("cpu")
    x1 = torch.randn(3, 5)
    torch.manual_seed(0)
    xt_1, v_1 = schedule.noise_pair(x1, torch.ones(3))
    torch.manual_seed(0)
    xt_0, v_0 = schedule.noise_pair(x1, torch.zeros(3))
    torch.testing.assert_close(xt_1, x1)
    # v = x1 - x0_noise → xt at t=0 is exactly x0_noise = x1 - v_0
    torch.testing.assert_close(xt_0, x1 - v_0)


def test_fm_zero_velocity_keeps_input():
    """Sample with a zero-velocity model returns the initial Gaussian noise."""
    schedule = FlowMatchingSchedule(FlowMatchingConfig()).to("cpu")
    torch.manual_seed(42)
    out = schedule.sample(shape=(2, 4), model_fn=_identity_model, steps=8)
    torch.manual_seed(42)
    expected = torch.randn(2, 4)
    torch.testing.assert_close(out, expected)


def test_fm_apply_ot_permutes_noise_to_match_data():
    """`_apply_ot` must reorder noise so that row ``i`` is the OT match for ``data[i]``.

    Builds a batch where each noise row coincides with exactly one (permuted) data
    row; the unique optimal assignment is to un-permute the noise, which makes the
    reordered output equal to ``data``.
    """
    pytest.importorskip("scipy")
    schedule = FlowMatchingSchedule(FlowMatchingConfig(minibatch_ot=True)).to("cpu")
    data = torch.tensor([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0], [40.0, 0.0]])
    noise = data[torch.tensor([2, 0, 3, 1])]
    reordered = schedule._apply_ot(noise, data)
    torch.testing.assert_close(reordered, data)


# ---------------------------------------------------------------------------
# FlowMatching joint variants (used by the AB-UPT diffusion trainer)
# ---------------------------------------------------------------------------


def _joint_zero_model(xt_list, t, condition):
    return [torch.zeros_like(x) for x in xt_list]


def test_fm_joint_training_losses_shared_t():
    schedule = FlowMatchingSchedule(FlowMatchingConfig()).to("cpu")
    x0_list = [torch.randn(3, 5), torch.randn(3, 7)]
    loss = schedule.training_losses_joint(x0_list, _joint_zero_model)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_fm_joint_training_losses_rejects_mismatched_batch():
    schedule = FlowMatchingSchedule(FlowMatchingConfig()).to("cpu")
    with pytest.raises(AssertionError, match="must share batch dim"):
        schedule.training_losses_joint(
            [torch.randn(3, 5), torch.randn(4, 7)],
            _joint_zero_model,
        )


def test_fm_sample_joint_shapes():
    schedule = FlowMatchingSchedule(FlowMatchingConfig()).to("cpu")
    samples = schedule.sample_joint(
        shapes=[(2, 5), (2, 7)],
        model_fn=_joint_zero_model,
        steps=4,
    )
    assert len(samples) == 2
    assert samples[0].shape == (2, 5)
    assert samples[1].shape == (2, 7)
