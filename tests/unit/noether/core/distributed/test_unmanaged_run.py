#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import os
from unittest.mock import MagicMock, patch

import pytest

from noether.core.distributed.run.unmanaged import _run_multiprocess, run_unmanaged

_MODULE_PATH = "noether.core.distributed.run.unmanaged"


@patch(_MODULE_PATH + ".parse_devices")
@patch(_MODULE_PATH + ".check_single_device_visible")
@patch(_MODULE_PATH + ".log_device_info")
@patch(_MODULE_PATH + ".accelerator_to_device")
@patch("torch.multiprocessing.spawn")
class TestRunUnmanagedDispatch:
    def test_single_process_gpu(
        self,
        mock_spawn,
        mock_acc_to_dev,
        mock_log,
        mock_check,
        mock_parse,
    ):
        mock_parse.return_value = (1, ["3"])  # world_size=1, device="3"
        mock_main = MagicMock()
        mock_acc_to_dev.return_value = "cuda:0"

        with patch.dict(os.environ, {}, clear=True):
            run_unmanaged(main=mock_main, devices="3", accelerator="gpu")

            assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
            mock_check.assert_called_once_with(accelerator="gpu")
            mock_log.assert_called_once()
            mock_acc_to_dev.assert_called_once_with(accelerator="gpu")

            mock_main.assert_called_once_with(device="cuda:0")
            mock_spawn.assert_not_called()

    def test_multi_process_spawn(
        self,
        mock_spawn,
        mock_acc_to_dev,
        mock_log,
        mock_check,
        mock_parse,
    ):
        mock_parse.return_value = (2, ["0", "1"])  # world_size=2
        mock_main = MagicMock()

        run_unmanaged(main=mock_main, devices="0,1", accelerator="gpu", master_port=12345)

        mock_spawn.assert_called_once()

        call_args = mock_spawn.call_args
        assert call_args.kwargs["nprocs"] == 2

        # args passed to _run_multiprocess: (accelerator, device_ids, master_port, world_size, main)
        fn_args = call_args.kwargs["args"]
        assert fn_args[0] == "gpu"
        assert fn_args[1] == ["0", "1"]
        assert fn_args[2] == 12345
        assert fn_args[3] == 2
        assert fn_args[4] == mock_main

    def test_multi_process_missing_port_raises(
        self,
        mock_spawn,
        mock_acc_to_dev,
        mock_log,
        mock_check,
        mock_parse,
    ):
        """Should raise RuntimeError if master_port is missing for multi-process."""
        mock_parse.return_value = (2, ["0", "1"])

        with pytest.raises(RuntimeError, match="master_port must be specified"):
            run_unmanaged(main=MagicMock(), devices="0,1", master_port=None)


@patch("torch.distributed.init_process_group")
@patch("torch.distributed.destroy_process_group")
@patch(_MODULE_PATH + ".check_single_device_visible")
@patch(_MODULE_PATH + ".get_backend")
@patch(_MODULE_PATH + ".accelerator_to_device")
class TestMultiprocessWorker:
    def test_worker_setup_execution(
        self,
        mock_acc,
        mock_backend,
        mock_check,
        mock_destroy,
        mock_init,
    ):
        rank = 1
        accelerator = "gpu"
        device_ids = ["0", "1"]
        master_port = 9999
        world_size = 2
        mock_main = MagicMock()

        mock_backend.return_value = "nccl"
        mock_acc.return_value = "cuda:0"

        with patch.dict(os.environ, {}, clear=True):
            _run_multiprocess(rank, accelerator, device_ids, master_port, world_size, mock_main)

            assert os.environ["MASTER_ADDR"] == "localhost"
            assert os.environ["MASTER_PORT"] == "9999"
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"

            mock_init.assert_called_once_with(backend="nccl", init_method="env://", world_size=2, rank=1)

            mock_main.assert_called_once_with(device="cuda:0")
            mock_destroy.assert_called_once()
