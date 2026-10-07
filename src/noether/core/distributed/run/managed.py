#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import atexit
import datetime
import logging
import os
import platform
import subprocess

import torch
from torch.distributed import barrier, destroy_process_group, init_process_group

from noether.core.distributed.config import (
    get_local_rank,
    get_managed_rank,
    get_managed_world_size,
    get_num_nodes,
    is_managed,
)
from noether.core.distributed.utils import accelerator_to_device, get_backend

logger = logging.getLogger(__name__)


def first_hostname_from_nodelist(nodelist: str) -> str:
    """Return the first concrete hostname from a (possibly compressed) SLURM nodelist.

    ``SLURM_JOB_NODELIST`` uses SLURM's *compressed* notation for multi-node jobs, e.g.
    ``node-[079,081]`` or ``node[01-04]``. A naive ``nodelist.split(",")[0]`` therefore
    yields a broken hostname such as ``node-[079`` that does not resolve, which makes the
    ``torch.distributed`` rendezvous time out. ``scontrol show hostnames`` expands the list
    to one real hostname per line; the first is used as the rendezvous master.

    Args:
        nodelist: The value of ``SLURM_JOB_NODELIST`` (compressed or plain).

    Returns:
        The first hostname in the (expanded) nodelist.

    Raises:
        RuntimeError: If the nodelist is compressed but ``scontrol`` is unavailable, so no
            concrete hostname can be derived.
    """
    try:
        result = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist],
            capture_output=True,
            text=True,
            check=True,
        )
        hostnames = result.stdout.split()
        if hostnames:
            return hostnames[0]
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        logger.warning(
            f"`scontrol show hostnames {nodelist}` failed ({exc}); falling back to parsing SLURM_JOB_NODELIST directly."
        )

    # Fallback: a plain, uncompressed list ("nodeA,nodeB") has its first host before the
    # first comma. A compressed list still contains '[' here, which we cannot expand
    # without scontrol -- fail loudly rather than return a bogus address.
    first = nodelist.split(",")[0]
    if "[" in first:
        raise RuntimeError(
            f"Cannot derive MASTER_ADDR from compressed SLURM_JOB_NODELIST={nodelist!r} "
            "because `scontrol` is unavailable. Set MASTER_ADDR explicitly in the environment."
        )
    return first


def run_managed(main, accelerator="gpu", devices=None):
    assert is_managed()
    # some HPCs dont set CUDA_VISIBLE_DEVICES at all (e.g. lux)
    device_id = None
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        local_rank = get_local_rank()
        if torch.cuda.device_count() > 1:
            assert torch.cuda.device_count() == int(os.environ.get("SLURM_NTASKS_PER_NODE", 1)), (
                f"Expected number of visible devices (torch.cuda.device_count()={torch.cuda.device_count()}) to match SLURM_NTASKS_PER_NODE ({os.environ.get('SLURM_NTASKS_PER_NODE', 1)}) when CUDA_VISIBLE_DEVICES is not set (local_rank={local_rank})"
            )
            torch.cuda.set_device(local_rank)  # set device to local_rank to initialize NCCL
            device_id = local_rank
    else:
        # srun doesnt set CUDA_VISIBLE_DEVICES
        split = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        assert len(split) == int(os.environ.get("SLURM_NTASKS_PER_NODE", 1)), (
            f"Expected number of visible devices (len(CUDA_VISIBLE_DEVICES)={len(split)}) to match SLURM_NTASKS_PER_NODE ({os.environ.get('SLURM_NTASKS_PER_NODE', 1)}) when CUDA_VISIBLE_DEVICES is set, make sure to set gpus_per_node"
        )
        if len(split) > 1:
            assert len(split) == int(os.environ.get("SLURM_NTASKS_PER_NODE", 1)), (
                f"Expected number of visible devices (len(CUDA_VISIBLE_DEVICES)={len(split)}) to match SLURM_NTASKS_PER_NODE ({os.environ.get('SLURM_NTASKS_PER_NODE', 1)}) when CUDA_VISIBLE_DEVICES is set"
            )
            local_rank = get_local_rank()
            torch.cuda.set_device(
                local_rank
            )  # set device to local_rank to initialize NCCL properly, even if multiple GPUs are visible
            device_id = local_rank
    assert devices is None, f"devices are set implicitly via environment (devices should be None but is '{devices}')"
    world_size = get_managed_world_size()
    if world_size == 1:
        # no need for setting up distributed stuff
        _run_managed_singleprocess(accelerator, main)
    else:
        # use all GPUs for training
        _run_managed_multiprocess(accelerator, main, device_id=device_id)


def _run_managed_singleprocess(accelerator, main):
    # single process
    logger.info("running single process slurm training")
    device = accelerator_to_device(accelerator)
    main(device=device)


def _run_managed_multiprocess(accelerator, main, device_id=None):
    # setup MASTER_ADDR & MASTER_PORT
    if not os.environ.get("MASTER_ADDR", ""):
        if "SLURM_JOB_NODELIST" in os.environ:
            os.environ["MASTER_ADDR"] = first_hostname_from_nodelist(os.environ["SLURM_JOB_NODELIST"])
        else:
            raise RuntimeError("SLURM_JOB_NODELIST not found in environment, cannot set MASTER_ADDR")
    if not os.environ.get("MASTER_PORT", ""):
        if "SLURM_JOB_ID" in os.environ:
            # derive a port from the slurm job id
            slurm_job_id = int(os.environ["SLURM_JOB_ID"])
            master_port = 15000 + (slurm_job_id % 10000)
            os.environ["MASTER_PORT"] = str(master_port)
            logger.info(f"setting MASTER_PORT={master_port} derived from SLURM_JOB_ID={slurm_job_id}")
        else:
            raise RuntimeError("SLURM_JOB_ID not found in environment, cannot set MASTER_PORT")
    world_size = get_managed_world_size()

    if not os.environ.get("WORLD_SIZE"):
        os.environ["WORLD_SIZE"] = str(world_size)
    rank = get_managed_rank()
    if not os.environ.get("RANK"):
        os.environ["RANK"] = str(rank)

    distributed_timeout_s = int(os.environ.get("DISTRIBUTED_TIMEOUT_S", "120"))

    # init process group
    logger.info(
        f"initializing rank={rank} local_rank={get_local_rank()} "
        f"nodes={get_num_nodes()} hostname={platform.uname().node} "
        f"master_addr={os.environ['MASTER_ADDR']} master_port={os.environ['MASTER_PORT']} "
        f"(waiting for all {world_size} processes to connect)"
        f"timeout={distributed_timeout_s}s",
    )
    init_process_group(
        backend=get_backend(accelerator),
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=distributed_timeout_s),
        device_id=torch.device(f"cuda:{device_id}") if accelerator == "gpu" and device_id is not None else None,
    )
    barrier()

    # start main_single
    device = accelerator_to_device(accelerator)
    atexit.register(destroy_process_group)  # ensure that process group is destroyed on exit, even if main() crashes
    main(device=device)

    destroy_process_group()
