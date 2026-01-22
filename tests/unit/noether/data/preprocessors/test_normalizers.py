#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.core.schemas.normalizers import (
    MeanStdNormalizerConfig,
    PositionNormalizerConfig,
    ShiftAndScaleNormalizerConfig,
)
from noether.data.preprocessors.normalizers import (
    MeanStdNormalization,
    PositionNormalizer,
    ShiftAndScaleNormalizer,
)


def test_init_with_scalars():
    """Tests the initialization of MeanStdNormalization with scalar mean and std."""
    mean = 10.0
    std = 2.0
    normalizer = MeanStdNormalization(MeanStdNormalizerConfig(mean=mean, std=std), normalization_key="test_normalizer")

    expected_shift = torch.tensor(-mean)
    expected_scale = torch.tensor(1.0 / (std + 1e-6))

    assert torch.allclose(normalizer.shift, expected_shift)
    assert torch.allclose(normalizer.scale, expected_scale)

    test_tensor = torch.tensor([12.0, 14.0])
    normalized_tensor = normalizer(test_tensor)
    expected_normalized = (test_tensor + expected_shift) * expected_scale
    assert torch.allclose(normalized_tensor, expected_normalized)
    assert torch.allclose(normalizer.denormalize(normalized_tensor), test_tensor, atol=1e-6)


def test_init_with_sequences():
    """Tests the initialization of MeanStdNormalization with sequence mean and std."""
    mean = [10.0, 20.0]
    std = [2.0, 4.0]
    normalizer = MeanStdNormalization(MeanStdNormalizerConfig(mean=mean, std=std), normalization_key="test_normalizer")

    expected_shift = torch.tensor([-10.0, -20.0])
    expected_scale = torch.tensor([1.0 / (2.0 + 1e-6), 1.0 / (4.0 + 1e-6)])

    assert torch.allclose(normalizer.shift, expected_shift)
    assert torch.allclose(normalizer.scale, expected_scale)

    test_tensor = torch.tensor([[12.0, 24.0], [14.0, 28.0]])
    normalized_tensor = normalizer(test_tensor)
    expected_normalized = (test_tensor + expected_shift) * expected_scale
    assert torch.allclose(normalized_tensor, expected_normalized)
    test_tensor = torch.rand((10, 10, 2))
    normalized_tensor = normalizer(test_tensor)
    expected_normalized = (test_tensor + expected_shift) * expected_scale
    assert torch.allclose(normalized_tensor, expected_normalized)
    assert torch.allclose(normalizer.denormalize(normalized_tensor), test_tensor, atol=1e-6)


def test_init_raises_value_error_for_mismatched_lengths():
    """Tests that a ValueError is raised if mean and std have different lengths."""
    mean = [10.0, 20.0]
    std = [2.0]
    with pytest.raises(ValueError, match="mean and std must have the same shape."):
        MeanStdNormalization(MeanStdNormalizerConfig(mean=mean, std=std))


def test_init_raises_value_error_for_negative_std():
    """Tests that a ValueError is raised if std contains negative values."""
    mean = [10.0, 20.0]
    std = [2.0, -4.0]
    with pytest.raises(ValueError, match="std must not contain negative values."):
        MeanStdNormalization(MeanStdNormalizerConfig(mean=mean, std=std), normalization_key="test_normalizer")


def test_init_raises_value_error_for_negative_std_close_to_zero():
    """Tests that a ValueError is raised for a small negative std."""
    mean = 10.0
    std = -1e-5  # This will be negative even after adding epsilon
    with pytest.raises(ValueError, match="std must not contain negative values."):
        MeanStdNormalization(MeanStdNormalizerConfig(mean=mean, std=std))


class TestPositionNormalizerInit:
    @pytest.mark.parametrize(
        ("raw_pos_min", "raw_pos_max", "scale", "expected_shift", "expected_scale"),
        [
            (
                [-10.0, -20.0],
                [10.0, 20.0],
                100,
                torch.tensor([10.0, 20.0]),
                torch.tensor([5.0, 2.5]),
            ),
            (
                torch.tensor([-10.0]),
                torch.tensor([10.0]),
                1000,
                torch.tensor([10.0]),
                torch.tensor([50.0]),
            ),
            (
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 4.0],
                1.0,
                torch.tensor([0.0, 0.0, 0.0]),
                torch.tensor([1.0, 0.5, 0.25]),
            ),
        ],
    )
    def test_init_calculates_shift_and_scale_correctly(
        self, raw_pos_min, raw_pos_max, scale, expected_shift, expected_scale
    ):
        """Tests that the shift and scale are calculated correctly during initialization."""
        normalizer = PositionNormalizer(
            PositionNormalizerConfig(raw_pos_min=raw_pos_min, raw_pos_max=raw_pos_max, scale=scale),
            normalization_key="test_key",
        )

        assert torch.allclose(normalizer.shift, expected_shift)
        assert torch.allclose(normalizer.scale, expected_scale)

        if isinstance(raw_pos_min, torch.Tensor):
            expected_min = raw_pos_min.clone().detach().float()
        else:
            expected_min = torch.tensor(raw_pos_min).float()

        if isinstance(raw_pos_max, torch.Tensor):
            expected_max = raw_pos_max.clone().detach().float()
        else:
            expected_max = torch.tensor(raw_pos_max).float()

        assert torch.allclose(normalizer.raw_pos_min.clone().detach(), expected_min)
        assert torch.allclose(normalizer.raw_pos_max.clone().detach(), expected_max)

        assert torch.allclose(normalizer.resizing_scale, torch.tensor(float(scale)))

    def test_init_with_default_scale(self):
        """Tests initialization with the default scale value."""
        raw_pos_min = [-10.0]
        raw_pos_max = [10.0]
        normalizer = PositionNormalizer(
            PositionNormalizerConfig.model_validate(dict(raw_pos_min=raw_pos_min, raw_pos_max=raw_pos_max)),
            normalization_key="test_key",
        )

        expected_scale = 1000.0 / (10.0 - (-10.0))  # 1000 / 20 = 50
        assert normalizer.scale.item() == pytest.approx(expected_scale)
        assert normalizer.resizing_scale.item() == 1000.0

        test_tensor = torch.tensor([-10.0, 0.0, 10.0])
        normalized_tensor = normalizer(test_tensor)
        assert torch.allclose(normalized_tensor, torch.tensor([0.0, 500.0, 1000.0]))
        assert torch.allclose(normalizer.denormalize(normalized_tensor), test_tensor)

    @pytest.mark.parametrize(
        ("raw_pos_min", "raw_pos_max", "expected_message"),
        [
            (
                [-10.0],
                [10.0, 20.0],
                "raw_pos_min and raw_pos_max must have the same shape.",
            ),
            (
                [-10.0, -20.0],
                [-10.0, -20.0],
                "raw_pos_max must be element-wise greater than raw_pos_min.",
            ),
        ],
    )
    def test_init_raises_value_error_for_invalid_min_max(self, raw_pos_min, raw_pos_max, expected_message):
        """Tests that a ValueError is raised for inconsistent or equal min/max values."""
        with pytest.raises(ValueError, match=expected_message):
            PositionNormalizer(PositionNormalizerConfig(raw_pos_min=raw_pos_min, raw_pos_max=raw_pos_max))

    @pytest.mark.parametrize("scale", [0, -100])
    def test_init_raises_value_error_for_non_positive_scale(self, scale):
        """Tests that a ValueError is raised if the scale is not a positive number."""
        with pytest.raises(ValueError, match="Input should be greater than 0"):
            PositionNormalizer(
                PositionNormalizerConfig(raw_pos_min=[-10], raw_pos_max=[10], scale=scale), normalization_key="test_key"
            )

    @pytest.mark.parametrize(
        ("raw_pos_min", "raw_pos_max", "expected_message"),
        [
            (1.0, [1.0], "raw_pos_min must be a Sequence or a torch.Tensor."),
            ([1.0], 1.0, "raw_pos_max must be a Sequence or a torch.Tensor."),
            (None, [1.0], "raw_pos_min must be a Sequence or a torch.Tensor."),
            ([1.0], None, "raw_pos_max must be a Sequence or a torch.Tensor."),
        ],
    )
    def test_init_raises_type_error_for_invalid_types(self, raw_pos_min, raw_pos_max, expected_message):
        """Tests that a TypeError is raised for invalid input types for min/max."""
        with pytest.raises((TypeError, ValueError)):
            PositionNormalizer(
                PositionNormalizerConfig(raw_pos_min=raw_pos_min, raw_pos_max=raw_pos_max), normalization_key="test_key"
            )

    def test_call_raises_value_error_for_out_of_bounds(self):
        """Tests that a ValueError is raised if input tensor has values outside the defined min/max range."""
        normalizer = PositionNormalizer(
            PositionNormalizerConfig(raw_pos_min=[0.0], raw_pos_max=[10.0]), normalization_key="test_key"
        )

        test_tensor = torch.tensor([-1.0, 5.0, 11.0])
        with pytest.raises(ValueError):
            normalizer(test_tensor)


def test_init_with_tensors():
    """Tests that the normalizer can be initialized with tensors for shift and scale."""
    shift = torch.tensor([1.0, 2.0])
    scale = torch.tensor([3.0, 4.0])
    normalizer = ShiftAndScaleNormalizer(
        ShiftAndScaleNormalizerConfig(shift=shift, scale=scale), normalization_key="test_normalizer"
    )
    assert torch.equal(normalizer.shift, shift)
    assert torch.equal(normalizer.scale, scale)
    assert not normalizer.logscale


def test_init_with_logscale():
    """Tests that the normalizer can be initialized with logscale parameters."""
    shift = [0.1, 0.2]
    scale = [0.3, 0.4]
    normalizer = ShiftAndScaleNormalizer(
        ShiftAndScaleNormalizerConfig(shift=shift, scale=scale, logscale=True),
        normalization_key="test_logscale_normalizer",
    )
    assert torch.equal(normalizer.shift, torch.tensor(shift, dtype=torch.float32))
    assert torch.equal(normalizer.scale, torch.tensor(scale, dtype=torch.float32))
    assert normalizer.logscale


@pytest.mark.parametrize(
    ("shift", "scale"),
    [
        ([1.0], None),
        (None, [1.0]),
        (torch.tensor([1.0]), None),
        (None, torch.tensor([1.0])),
    ],
)
def test_init_raises_value_error_if_only_one_of_shift_scale_provided(shift, scale):
    """Tests that a ValueError is raised if only one of shift or scale is provided."""
    with pytest.raises(ValueError, match="Could not convert None to torch.Tensor"):
        ShiftAndScaleNormalizer(
            ShiftAndScaleNormalizerConfig(shift=shift, scale=scale), normalization_key="test_normalizer"
        )


@pytest.mark.parametrize(
    ("shift", "scale", "err_type", "match"),
    [
        (1, [1.0], (TypeError, ValueError), "shift and scale must have the same shape."),
        ([1.0], 1, (TypeError, ValueError), "shift and scale must have the same shape."),
        ([1.0], [1.0, 2.0], ValueError, "shift and scale must have the same shape."),
    ],
)
def test_init_raises_error_for_invalid_shift_scale(shift, scale, err_type, match):
    """Tests that an error is raised for invalid shift/scale inputs."""
    with pytest.raises(err_type, match=match):
        ShiftAndScaleNormalizer(
            ShiftAndScaleNormalizerConfig(shift=shift, scale=scale), normalization_key="test_normalizer"
        )


def test_init_raises_type_error_for_non_boolean_logscale():
    """Tests that a TypeError is raised if logscale is not a boolean."""
    with pytest.raises(ValueError):
        ShiftAndScaleNormalizer(
            ShiftAndScaleNormalizerConfig(
                shift=[0.1],
                scale=[0.2],
                logscale="Truer",
            ),
            normalization_key="test_normalizer",
        )


@pytest.mark.parametrize(
    ("shift", "scale"),
    [
        ([1.0], None),
        (None, [1.0]),
    ],
)
def test_init_raises_value_error_for_missing_logscale_params(shift, scale):
    """Tests that a ValueError is raised if logscale is True but params are missing."""
    with pytest.raises(
        ValueError,
        match="Could not convert None to torch.Tensor",
    ):
        ShiftAndScaleNormalizer(
            ShiftAndScaleNormalizerConfig(shift=shift, scale=scale, logscale=True),
            normalization_key="test_logscale_normalizer",
        )


@pytest.mark.parametrize(
    ("shift", "scale", "err_type", "match"),
    [
        (
            1,
            [1.0],
            (TypeError, ValueError),
            "shift and scale must have the same shape.",
        ),
        (
            [1.0],
            1,
            (TypeError, ValueError),
            "shift and scale must have the same shape.",
        ),
        (
            [1.0],
            [1.0, 2.0],
            ValueError,
            "shift and scale must have the same shape.",
        ),
    ],
)
def test_init_raises_error_for_invalid_logscale_params(shift, scale, err_type, match):
    """Tests that an error is raised for invalid logscale parameter inputs."""
    with pytest.raises(err_type, match=match):
        ShiftAndScaleNormalizer(
            ShiftAndScaleNormalizerConfig(
                shift=shift,
                scale=scale,
                logscale=True,
            ),
            normalization_key="test_logscale_normalizer",
        )


def test_call_raises_value_error_for_out_of_bounds_input():
    """Tests that a ValueError is raised if input tensor has incorrect dimensions."""
    normalizer = PositionNormalizer(
        PositionNormalizerConfig(raw_pos_min=[1.0, 0.0], raw_pos_max=[2.0, 1.0]), normalization_key="test_normalizer"
    )
    with pytest.raises(ValueError):
        normalizer(torch.tensor([[0.0, 2.0], [3.0, 4.0]]))  # type: ignore[arg-type]
