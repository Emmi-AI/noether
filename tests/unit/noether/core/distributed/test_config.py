#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import os
from unittest.mock import MagicMock, patch

import pytest

from noether.core.distributed import config as config_module
from noether.core.distributed.config import DistributedConfig, set_config


class TestDistributedConfig:
    @pytest.fixture
    def config(self):
        return DistributedConfig()

    @pytest.fixture
    def mock_dist(self):
        with patch("noether.core.distributed.config.dist") as mock:
            yield mock

    # --- SLURM & Environment Tests ---

    def test_is_managed_slurm(self, config):
        with patch.dict(os.environ, {"SLURM_PROCID": "0"}):
            assert config.is_managed() is True

    def test_is_managed_false(self, config):
        with patch.dict(os.environ, {}, clear=True):
            assert config.is_managed() is False

    def test_get_local_rank_slurm(self, config):
        with patch.dict(os.environ, {"SLURM_LOCALID": "3"}):
            assert config.get_local_rank() == 3

    def test_get_local_rank_fallback(self, config, mock_dist):
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 5

        with patch.dict(os.environ, {}, clear=True):
            # Should fall back to self.get_rank() -> dist.get_rank()
            assert config.get_local_rank() == 5

    def test_get_num_nodes(self, config):
        with patch.dict(os.environ, {"SLURM_JOB_NUM_NODES": "4"}):
            assert config.get_num_nodes() == 4

        with patch.dict(os.environ, {}, clear=True):
            assert config.get_num_nodes() == 1

    def test_get_managed_world_size(self, config):
        env_vars = {
            "SLURM_JOB_NUM_NODES": "2",
            "SLURM_NTASKS_PER_NODE": "4",
        }
        with patch.dict(os.environ, env_vars):
            # 2 nodes * 4 tasks = 8:
            assert config.get_managed_world_size() == 8

    def test_get_managed_rank(self, config):
        with patch.dict(os.environ, {"SLURM_PROCID": "10"}):
            assert config.get_managed_rank() == 10

    def test_get_managed_rank_error(self, config):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="SLURM_PROCID not found"):
                config.get_managed_rank()

    # --- Torch Distributed Wrappers Tests ---

    def test_is_distributed(self, config, mock_dist):
        # distributed is active
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True
        assert config.is_distributed() is True

        # distributed not initialized
        mock_dist.is_initialized.return_value = False
        assert config.is_distributed() is False

    def test_get_rank_and_world_size(self, config, mock_dist):
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 7
        mock_dist.get_world_size.return_value = 16

        assert config.get_rank() == 7
        assert config.get_world_size() == 16

        # inactive distributed (defaults):
        mock_dist.is_initialized.return_value = False
        assert config.get_rank() == 0
        assert config.get_world_size() == 1

    def test_barrier(self, config, mock_dist):
        mock_dist.is_available.return_value = True

        # Should call dist.barrier if initialized:
        mock_dist.is_initialized.return_value = True
        config.barrier()
        mock_dist.barrier.assert_called_once()

        # Should NOT call dist.barrier if not initialized:
        mock_dist.barrier.reset_mock()
        mock_dist.is_initialized.return_value = False
        config.barrier()
        mock_dist.barrier.assert_not_called()

    # --- Data Rank Logic Tests ---

    def test_is_rank0_flags(self, config):
        with patch.object(config, "get_rank", return_value=0):
            assert config.is_rank0() is True
        with patch.object(config, "get_rank", return_value=1):
            assert config.is_rank0() is False

        with patch.object(config, "get_local_rank", return_value=0):
            assert config.is_local_rank0() is True
        with patch.object(config, "get_local_rank", return_value=1):
            assert config.is_local_rank0() is False

    def test_is_data_rank0(self, config):
        # Scenario 1: multi-GPU Distributed, rank 0 on node
        # local_rank=0, world_size=4 -> True
        with patch.object(config, "get_local_rank", return_value=0):
            with patch.object(config, "get_world_size", return_value=4):
                assert config.is_data_rank0() is True

        # Scenario 2: multi-GPU distributed, rank 1 on node
        # local_rank=1, world_size=4 -> False
        with patch.object(config, "get_local_rank", return_value=1):
            with patch.object(config, "get_world_size", return_value=4):
                assert config.is_data_rank0() is False

        # Scenario 3: single process (or non-distributed SLURM array job)
        # local_rank=0 (default), world_size=1 -> True
        with patch.object(config, "get_local_rank", return_value=0):
            with patch.object(config, "get_world_size", return_value=1):
                assert config.is_data_rank0() is True

        # Scenario 4: "weird" case - local_rank 1 but world_size 1.
        # This technically shouldn't happen in standard setups, but logic dictates True.
        with patch.object(config, "get_local_rank", return_value=1):
            with patch.object(config, "get_world_size", return_value=1):
                assert config.is_data_rank0() is True


def test_global_function_delegation():
    mock_config = MagicMock()

    original_config = config_module._config

    try:
        set_config(mock_config)

        config_module.is_managed()
        config_module.get_rank()
        config_module.barrier()
        config_module.get_local_rank()
        config_module.get_num_nodes()
        config_module.get_managed_world_size()
        config_module.get_managed_rank()
        config_module.is_distributed()
        config_module.get_world_size()
        config_module.is_data_rank0()
        config_module.is_rank0()
        config_module.is_local_rank0()

        mock_config.is_managed.assert_called_once()
        mock_config.get_rank.assert_called_once()
        mock_config.barrier.assert_called_once()
        mock_config.get_local_rank.assert_called_once()
        mock_config.get_num_nodes.assert_called_once()
        mock_config.get_managed_world_size.assert_called_once()
        mock_config.get_managed_rank.assert_called_once()
        mock_config.is_distributed.assert_called_once()
        mock_config.get_world_size.assert_called_once()
        mock_config.is_data_rank0.assert_called_once()
        mock_config.is_rank0.assert_called_once()
        mock_config.is_local_rank0.assert_called_once()
    finally:
        set_config(original_config)
