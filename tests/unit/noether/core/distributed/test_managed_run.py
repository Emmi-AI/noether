#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import os
from unittest.mock import MagicMock, patch

import pytest

from noether.core.distributed.run.managed import (
    _run_managed_multiprocess,
    first_hostname_from_nodelist,
    run_managed,
)

_MODULE_PATH = "noether.core.distributed.run.managed"


@patch(_MODULE_PATH + ".is_managed")
@patch(_MODULE_PATH + ".get_managed_world_size")
@patch(_MODULE_PATH + ".get_local_rank")
@patch(_MODULE_PATH + ".accelerator_to_device")
@patch(_MODULE_PATH + "._run_managed_singleprocess")
@patch(_MODULE_PATH + "._run_managed_multiprocess")
@patch(_MODULE_PATH + ".torch.cuda.device_count")
@patch(_MODULE_PATH + ".torch.cuda.set_device")
class TestRunManagedDispatch:
    def test_not_managed_raises(self, mock_set_device, mock_device_count, mock_multi, mock_single, *args):
        with patch(_MODULE_PATH + ".is_managed", return_value=False):
            with pytest.raises(AssertionError):
                run_managed(MagicMock())

    def test_devices_set_raises(self, mock_set_device, mock_device_count, *args):
        with patch(_MODULE_PATH + ".is_managed", return_value=True):
            with pytest.raises(AssertionError, match="devices should be None"):
                with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
                    run_managed(MagicMock(), devices=4)

    def test_cuda_env_missing_sets_local_rank(
        self,
        mock_set_device,
        mock_device_count,
        mock_multi,
        mock_single,
        mock_acc_to_dev,
        mock_local_rank,
        mock_world_size,
        mock_is_managed,
    ):
        """If CUDA_VISIBLE_DEVICES is missing, `torch.cuda.set_device` should be called with local_rank."""
        mock_is_managed.return_value = True
        mock_local_rank.return_value = 3
        mock_world_size.return_value = 1
        mock_device_count.return_value = 2

        with patch.dict(os.environ, {"SLURM_NTASKS_PER_NODE": "2"}, clear=True):
            run_managed(MagicMock())
            mock_set_device.assert_called_once_with(3)

    def test_cuda_env_list_picks_correct_device(
        self,
        mock_set_device,
        mock_device_count,
        mock_multi,
        mock_single,
        mock_acc_to_dev,
        mock_local_rank,
        mock_world_size,
        mock_is_managed,
    ):
        """If multiple devices are visible, `torch.cuda.set_device` should pick the local_rank."""
        mock_is_managed.return_value = True
        mock_local_rank.return_value = 1
        mock_world_size.return_value = 1

        # Simulating srun allocating 4 GPUs to the node:
        initial_env = {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "SLURM_NTASKS_PER_NODE": "4"}

        with patch.dict(os.environ, initial_env, clear=True):
            run_managed(MagicMock())
            mock_set_device.assert_called_once_with(1)

    def test_dispatch_single_process(
        self,
        mock_set_device,
        mock_device_count,
        mock_multi,
        mock_single,
        mock_acc_to_dev,
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
        mock_set_device,
        mock_device_count,
        mock_multi,
        mock_single,
        mock_acc_to_dev,
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

            mock_init.assert_called_once()
            called_kwargs = mock_init.call_args.kwargs
            assert called_kwargs["backend"] == "nccl"
            assert called_kwargs["init_method"] == "env://"
            assert called_kwargs["world_size"] == 8
            assert called_kwargs["rank"] == 0
            mock_barrier.assert_called_once()
            mock_main.assert_called_once()
            mock_destroy.assert_called_once()

    def test_master_addr_expands_compressed_nodelist(
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
        """A multi-node SLURM_JOB_NODELIST is compressed (e.g. node-[01,02]); MASTER_ADDR must
        be the first *expanded* hostname, not the broken `node-[01` from a naive comma split.

        This is the regression test for the multi-node rendezvous timeout bug.
        """
        mock_world_size.return_value = 2
        mock_rank.return_value = 0
        mock_backend.return_value = "nccl"

        env_vars = {"SLURM_JOB_NODELIST": "node-[01,02]", "SLURM_JOB_ID": "1234"}
        scontrol_result = MagicMock(stdout="node-01\nnode-02\n")

        with (
            patch.dict(os.environ, env_vars, clear=True),
            patch(_MODULE_PATH + ".subprocess.run", return_value=scontrol_result) as mock_run,
        ):
            _run_managed_multiprocess(accelerator="gpu", main=MagicMock())

            mock_run.assert_called_once_with(
                ["scontrol", "show", "hostnames", "node-[01,02]"],
                capture_output=True,
                text=True,
                check=True,
            )
            assert os.environ["MASTER_ADDR"] == "node-01"

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


class TestFirstHostnameFromNodelist:
    def test_expands_compressed_nodelist_via_scontrol(self):
        scontrol_result = MagicMock(stdout="slurm-h100-rno-199-079\nslurm-h100-rno-199-081\n")
        with patch(_MODULE_PATH + ".subprocess.run", return_value=scontrol_result) as mock_run:
            assert first_hostname_from_nodelist("slurm-h100-rno-199-[079,081]") == "slurm-h100-rno-199-079"
            mock_run.assert_called_once()

    def test_fallback_to_plain_list_when_scontrol_missing(self):
        """If `scontrol` is unavailable, a plain comma-separated list still resolves."""
        with patch(_MODULE_PATH + ".subprocess.run", side_effect=FileNotFoundError):
            assert first_hostname_from_nodelist("node-01,node-02") == "node-01"

    def test_compressed_without_scontrol_raises(self):
        """A compressed list cannot be expanded without scontrol -- fail loudly, never return
        a bogus `node-[01` address."""
        with patch(_MODULE_PATH + ".subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(RuntimeError, match="compressed SLURM_JOB_NODELIST"):
                first_hostname_from_nodelist("node-[01-04]")
