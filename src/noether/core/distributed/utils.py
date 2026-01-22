#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
import os

import torch
import yaml

logger = logging.getLogger(__name__)


def check_single_device_visible(accelerator):
    if accelerator == "cpu":
        # nothing to check
        return
    elif accelerator == "gpu":
        # if "import torch" is called before "CUDA_VISIBLE_DEVICES" is set, torch will see all devices
        assert "CUDA_VISIBLE_DEVICES" in os.environ
        assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1, f"{os.environ['CUDA_VISIBLE_DEVICES']}"
        import torch

        assert torch.cuda.is_available(), "CUDA not available use --accelerator cpu to run on cpu"
        visible_device_count = torch.cuda.device_count()
        assert visible_device_count <= 1, (
            "set CUDA_VISIBLE_DEVICES before importing torch "
            f"CUDA_VISIBLE_DEVICES='{os.environ['CUDA_VISIBLE_DEVICES']}' "
            f"torch.cuda.device_count={visible_device_count}"
        )
    elif accelerator == "mps":
        import torch

        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS selected but not available, use --accelerator cpu to run on cpu")
        return torch.mps.device_count() == 1
    else:
        raise NotImplementedError


def get_backend(accelerator, device_ids=None):
    if accelerator == "cpu":
        # gloo is recommended for cpu multiprocessing
        # https://pytorch.org/docs/stable/distributed.html#which-backend-to-use
        return "gloo"
    if os.name == "nt":
        # windows doesn't support nccl
        return "gloo"
    # MIG doesn't support NCCL
    if device_ids is not None:
        for device_id in device_ids:
            try:
                int(device_id)
            except ValueError:
                return "gloo"
    # nccl is recommended for gpu multiprocessing
    # https://pytorch.org/docs/stable/distributed.html#which-backend-to-use
    return "nccl"


def accelerator_to_device(accelerator):
    ACCELERATOR_DEVICE_MAP = {"cpu": "cpu", "gpu": "cuda", "mps": "mps"}
    if accelerator not in ACCELERATOR_DEVICE_MAP:
        raise NotImplementedError(f"Unsupported accelerator: {accelerator}")

    return ACCELERATOR_DEVICE_MAP[accelerator]


def parse_devices(accelerator: str, devices: str | None):
    # GPU + no explicit devices -> use all visible GPUs on node
    if accelerator == "gpu" and devices is None:
        logger.info("no device subset defined -> use all devices on node")
        count = torch.cuda.device_count()
        devices_ids = [str(idx) for idx in range(count)]
        return count, devices_ids

    # If devices is None for CPU (or any other accelerator), default to single "0":
    if devices is None:
        return 1, ["0"]

    # Parse explicit device string for BOTH CPU and GPU:
    try:
        device_ids = yaml.safe_load(f"[{devices}]")
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid devices specification '{devices}' (specify devices like '0' or '0,1,2')") from exc

    if not isinstance(device_ids, list) or not all(isinstance(d, int) for d in device_ids):
        raise ValueError(f"invalid devices specification '{devices}' (specify multiple devices like '0,1,2,3')")

    # always return strings
    device_ids = [str(d) for d in device_ids]
    return len(device_ids), device_ids


def log_device_info(accelerator, device_ids):
    if accelerator == "cpu":
        for i in range(len(device_ids)):
            logger.debug(f"device {i}: cpu")
    elif accelerator == "gpu":
        # retrieve device names via nvidia-smi because CUDA_VISIBLE_DEVICES needs to be set before calling anything
        # in torch.cuda -> only 1 visible device
        all_devices = os.popen("nvidia-smi --query-gpu=gpu_name --format=csv,noheader").read().strip().split("\n")
        for i, device_id in enumerate(device_ids):
            try:
                device_id = int(device_id)
                logger.debug(f"device {i}: {all_devices[device_id]} (id={device_id})")
            except ValueError:
                # MIG device
                logger.info("Can't retrieve MIG device name via nvidia-smi")
    elif accelerator == "mps":
        import torch

        logger.info(f"MPS device count: {torch.mps.device_count()}")
    else:
        raise NotImplementedError
