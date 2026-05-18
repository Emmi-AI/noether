#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import pytest
import torch

from noether.core.schemas.normalizers import FieldNormalizerConfig
from noether.data.preprocessors.normalizers import FieldNormalizer

SAMPLE_STATISTICS = {
    "surface_pressure_mean": [10.0],
    "surface_pressure_std": [2.0],
    "volume_velocity_mean": [1.0, 2.0, 3.0],
    "volume_velocity_std": [0.5, 1.0, 1.5],
    "volume_vorticity_logscale_mean": [0.1, 0.2, 0.3],
    "volume_vorticity_logscale_std": [1.0, 2.0, 3.0],
    "raw_pos_min": [-10.0, -20.0, -30.0],
    "raw_pos_max": [10.0, 20.0, 30.0],
    "_zero": [0.0],
}


class TestFieldNormalizerConfig:
    def test_default_strategy_is_mean_std(self):
        config = FieldNormalizerConfig()
        assert config.strategy == "mean_std"

    def test_kind_points_to_field_normalizer(self):
        config = FieldNormalizerConfig()
        assert config.kind == "noether.data.preprocessors.normalizers.FieldNormalizer"

    def test_position_strategy(self):
        config = FieldNormalizerConfig(strategy="position", scale=500.0)
        assert config.strategy == "position"
        assert config.scale == 500.0

    def test_stat_keys_override(self):
        config = FieldNormalizerConfig(
            strategy="mean_std",
            stat_keys={"mean": "custom_mean", "std": "custom_std"},
        )
        assert config.stat_keys == {"mean": "custom_mean", "std": "custom_std"}

    def test_logscale_flag(self):
        config = FieldNormalizerConfig(strategy="mean_std", logscale=True)
        assert config.logscale is True


class TestFieldNormalizer:
    def test_mean_std_default_keys(self):
        config = FieldNormalizerConfig(strategy="mean_std")
        normalizer = FieldNormalizer(config, statistics=SAMPLE_STATISTICS, normalization_key="surface_pressure")

        test_tensor = torch.tensor([[10.0], [12.0]])
        result = normalizer(test_tensor)
        assert result.shape == test_tensor.shape
        # Denormalize should recover original
        assert torch.allclose(normalizer.denormalize(result), test_tensor, atol=1e-5)

    def test_mean_std_custom_stat_keys(self):
        config = FieldNormalizerConfig(
            strategy="mean_std",
            logscale=True,
            stat_keys={"mean": "volume_vorticity_logscale_mean", "std": "volume_vorticity_logscale_std"},
        )
        normalizer = FieldNormalizer(config, statistics=SAMPLE_STATISTICS, normalization_key="volume_vorticity")
        # Should not raise — correct keys are resolved
        assert normalizer is not None

    def test_mean_std_with_zero_key(self):
        stats = {
            **SAMPLE_STATISTICS,
            "_zero": [0.0, 0.0, 0.0],  # Must match dimensionality of std
        }
        config = FieldNormalizerConfig(
            strategy="mean_std",
            stat_keys={"mean": "_zero", "std": "volume_vorticity_logscale_std"},
        )
        normalizer = FieldNormalizer(config, statistics=stats, normalization_key="volume_vorticity")
        assert normalizer is not None

    def test_mean_std_scalar_stats(self):
        stats = {"scalar_field_mean": 5.0, "scalar_field_std": 1.5}
        config = FieldNormalizerConfig(strategy="mean_std")
        normalizer = FieldNormalizer(config, statistics=stats, normalization_key="scalar_field")

        test_tensor = torch.tensor([5.0, 6.5])
        result = normalizer(test_tensor)
        assert torch.allclose(normalizer.denormalize(result), test_tensor, atol=1e-5)

    def test_position_default_keys(self):
        config = FieldNormalizerConfig(strategy="position", scale=1000.0)
        normalizer = FieldNormalizer(
            config,
            statistics={"surface_position_min": [-10.0], "surface_position_max": [10.0]},
            normalization_key="surface_position",
        )

        test_tensor = torch.tensor([-10.0, 0.0, 10.0])
        result = normalizer(test_tensor)
        assert torch.allclose(result, torch.tensor([0.0, 500.0, 1000.0]))

    def test_position_custom_stat_keys(self):
        config = FieldNormalizerConfig(
            strategy="position",
            stat_keys={"min": "raw_pos_min", "max": "raw_pos_max"},
            scale=1000.0,
        )
        normalizer = FieldNormalizer(config, statistics=SAMPLE_STATISTICS, normalization_key="surface_position")
        assert normalizer is not None

    def test_position_custom_scale(self):
        config = FieldNormalizerConfig(strategy="position", scale=500.0)
        normalizer = FieldNormalizer(
            config,
            statistics={"volume_position_min": [0.0], "volume_position_max": [10.0]},
            normalization_key="volume_position",
        )

        test_tensor = torch.tensor([0.0, 5.0, 10.0])
        result = normalizer(test_tensor)
        assert torch.allclose(result, torch.tensor([0.0, 250.0, 500.0]))

    def test_position_zero_center(self):
        config = FieldNormalizerConfig(strategy="position", scale=500.0, zero_center=True)
        normalizer = FieldNormalizer(
            config,
            statistics={"volume_position_min": [0.0], "volume_position_max": [10.0]},
            normalization_key="volume_position",
        )

        test_tensor = torch.tensor([0.0, 5.0, 10.0])
        result = normalizer(test_tensor)
        assert torch.allclose(result, torch.tensor([-500.0, 0.0, 500.0]))

    def test_missing_stat_key_raises(self):
        config = FieldNormalizerConfig(strategy="mean_std")
        with pytest.raises(KeyError):
            FieldNormalizer(config, statistics=SAMPLE_STATISTICS, normalization_key="nonexistent_field")

    def test_unknown_strategy_raises(self):
        config = FieldNormalizerConfig.__new__(FieldNormalizerConfig)
        object.__setattr__(config, "strategy", "unknown")
        object.__setattr__(config, "stat_keys", None)
        object.__setattr__(config, "logscale", False)
        object.__setattr__(config, "scale", 1000.0)
        object.__setattr__(config, "kind", "noether.data.preprocessors.normalizers.FieldNormalizer")
        with pytest.raises(ValueError, match="Unknown normalizer type"):
            FieldNormalizer(config, statistics=SAMPLE_STATISTICS, normalization_key="test")

    def test_denormalize_roundtrip(self):
        config = FieldNormalizerConfig(strategy="mean_std")
        normalizer = FieldNormalizer(config, statistics=SAMPLE_STATISTICS, normalization_key="volume_velocity")

        test_tensor = torch.randn(5, 3)
        # Shift to be around the mean so values are reasonable
        test_tensor = test_tensor + torch.tensor([1.0, 2.0, 3.0])
        result = normalizer(test_tensor)
        recovered = normalizer.denormalize(result)
        assert torch.allclose(recovered, test_tensor, atol=1e-5)
