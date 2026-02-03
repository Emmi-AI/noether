#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import os
from unittest.mock import MagicMock, patch

import pytest

from noether.core.distributed.run.managed import _run_managed_multiprocess, run_managed

_MODULE_PATH = "noether.core.distributed.run.managed"


@patch(_MODULE_PATH + ".is_managed")
@patch(_MODULE_PATH + ".get_managed_world_size")
@patch(_MODULE_PATH + ".get_local_rank")
@patch(_MODULE_PATH + ".check_single_device_visible")
@patch(_MODULE_PATH + ".accelerator_to_device")
@patch(_MODULE_PATH + "._run_managed_singleprocess")
@patch(_MODULE_PATH + "._run_managed_multiprocess")
class TestRunManagedDispatch:
    def test_not_managed_raises(self, mock_multi, mock_single, *args):
        with patch(_MODULE_PATH + ".is_managed", return_value=False):
            with pytest.raises(AssertionError):
                run_managed(MagicMock())

    def test_devices_set_raises(self, *args):
        with patch(_MODULE_PATH + ".is_managed", return_value=True):
            with pytest.raises(AssertionError, match="devices should be None"):
                run_managed(MagicMock(), devices=4)

    def test_cuda_env_missing_sets_local_rank(
        self,
        mock_multi,
        mock_single,
        mock_acc_to_dev,
        mock_check,
        mock_local_rank,
        mock_world_size,
        mock_is_managed,
    ):
        """If CUDA_VISIBLE_DEVICES is missing, it should be set to local_rank."""
        mock_is_managed.return_value = True
        mock_local_rank.return_value = 3
        mock_world_size.return_value = 1

        with patch.dict(os.environ, {}, clear=True):
            run_managed(MagicMock())
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"

    def test_cuda_env_list_picks_correct_device(
        self,
        mock_multi,
        mock_single,
        mock_acc_to_dev,
        mock_check,
        mock_local_rank,
        mock_world_size,
        mock_is_managed,
    ):
        """If multiple devices are visible, it should isolate the one for the local_rank."""
        mock_is_managed.return_value = True
        mock_local_rank.return_value = 1
        mock_world_size.return_value = 1

        # Simulating srun allocating 4 GPUs to the node:
        initial_env = {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}

        with patch.dict(os.environ, initial_env, clear=True):
            run_managed(MagicMock())
            # Should be set to '1' (the device at index 1):
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"

    def test_dispatch_single_process(
        self,
        mock_multi,
        mock_single,
        mock_acc_to_dev,
        mock_check,
        mock_local_rank,
        mock_world_size,
        mock_is_managed,
    ):
        mock_is_managed.return_value = True
        mock_world_size.return_value = 1
        mock_main = MagicMock()

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            run_managed(mock_main)

        mock_single.assert_called_once()
        mock_multi.assert_not_called()

    def test_dispatch_multi_process(
        self,
        mock_multi,
        mock_single,
        mock_acc_to_dev,
        mock_check,
        mock_local_rank,
        mock_world_size,
        mock_is_managed,
    ):
        mock_is_managed.return_value = True
        mock_world_size.return_value = 4
        mock_main = MagicMock()

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            run_managed(mock_main)

        mock_multi.assert_called_once()
        mock_single.assert_not_called()


@patch(_MODULE_PATH + ".init_process_group")
@patch(_MODULE_PATH + ".destroy_process_group")
@patch(_MODULE_PATH + ".barrier")
@patch(_MODULE_PATH + ".get_managed_rank")
@patch(_MODULE_PATH + ".get_managed_world_size")
@patch(_MODULE_PATH + ".get_local_rank")
@patch(_MODULE_PATH + ".get_num_nodes")
@patch(_MODULE_PATH + ".get_backend")
@patch(_MODULE_PATH + ".accelerator_to_device")
class TestMultiProcessExecution:
    def test_master_addr_port_derivation(
        self,
        mock_acc,
        mock_backend,
        mock_nodes,
        mock_local_rank,
        mock_world_size,
        mock_rank,
        mock_barrier,
        mock_destroy,
        mock_init,
    ):
        mock_main = MagicMock()
        mock_world_size.return_value = 8
        mock_rank.return_value = 0
        mock_backend.return_value = "nccl"

        env_vars = {
            "SLURM_JOB_NODELIST": "node-01,node-02",
            "SLURM_JOB_ID": "1234",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            _run_managed_multiprocess(accelerator="gpu", main=mock_main)

            assert os.environ["MASTER_ADDR"] == "node-01"
            assert os.environ["MASTER_PORT"] == "16234"

            mock_init.assert_called_once_with(backend="nccl", init_method="env://", world_size=8, rank=0)
            mock_barrier.assert_called_once()
            mock_main.assert_called_once()
            mock_destroy.assert_called_once()

    def test_missing_slurm_nodelist_raises(self, *mocks):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="SLURM_JOB_NODELIST not found"):
                _run_managed_multiprocess(accelerator="gpu", main=MagicMock())

    def test_missing_slurm_jobid_raises(self, *mocks):
        env_vars = {"SLURM_JOB_NODELIST": "node-01"}
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(RuntimeError, match="SLURM_JOB_ID not found"):
                _run_managed_multiprocess(accelerator="gpu", main=MagicMock())

    def test_existing_master_addr_is_respected(
        self,
        mock_acc,
        mock_backend,
        mock_nodes,
        mock_local_rank,
        mock_world_size,
        mock_rank,
        mock_barrier,
        mock_destroy,
        mock_init,
    ):
        env_vars = {
            "MASTER_ADDR": "existing-master",
            "MASTER_PORT": "9999",
            # SLURM vars present but shouldn't be used for master/port:
            "SLURM_JOB_NODELIST": "node-01",
            "SLURM_JOB_ID": "1234",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            _run_managed_multiprocess(accelerator="gpu", main=MagicMock())

            assert os.environ["MASTER_ADDR"] == "existing-master"
            assert os.environ["MASTER_PORT"] == "9999"
