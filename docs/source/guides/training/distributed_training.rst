How to run distributed training jobs
====================================

Noether supports data-parallel (PyTorch DDP) training across multiple GPUs and multiple
nodes. There are two ways to launch a distributed run:

- **Unmanaged**: you run ``noether-train`` directly on a host with several
  GPUs. Noether spawns one worker process per device for you.
- **Managed (SLURM)**: SLURM launches one worker per GPU. You can either
  write your own ``sbatch`` script or use the
  :doc:`noether-train-submit-job <launch_job>` helper.

Whichever path you take, the same scaling rule applies to the trainer config:

.. important::

   ``trainer.effective_batch_size`` is the **global batch size** used for each
   gradient step (summed across all ranks). It must be a multiple of the total
   number of GPUs (``world_size``). Per-device batch size is
   ``effective_batch_size / world_size``; if that exceeds
   ``trainer.max_batch_size``, the trainer inserts gradient accumulation steps
   automatically.

   Example: 8 GPUs with ``effective_batch_size=64`` → each rank processes 8
   samples per gradient step.

Unmanaged: single node, multiple GPUs
-------------------------------------

On a workstation without SLURM, just run ``noether-train``. Noether
detects how many devices are visible to the process and spawns one worker per
device using ``torch.multiprocessing.spawn`` — no ``torchrun`` or external
launcher required.

Use all visible GPUs (default):

.. code-block:: bash

   uv run noether-train --hp configs/train_shapenet.yaml \
       +experiment/shapenet=upt \
       +accelerator=gpu \
       dataset_root=/home/user/data/shapenet_car

Pin to a subset of GPUs with ``devices``:

.. code-block:: bash

   uv run noether-train --hp configs/train_shapenet.yaml \
       +experiment/shapenet=upt \
       +accelerator=gpu \
       +devices=\"0,1,2,3\" \
       dataset_root=/home/user/data/shapenet_car

A few notes for the unmanaged path:

- All target devices must be visible to the launching process. If you set
  ``CUDA_VISIBLE_DEVICES`` outside Noether, only those GPUs are eligible; the
  ``devices`` override then selects from that masked set.
- Multi-process unmanaged runs are **single-node only** — ``MASTER_ADDR`` is
  hard-coded to ``localhost``. Use SLURM (or another launcher) for multi-node.
- The rendezvous port can be set via ``master_port`` in the config or the
  ``MASTER_PORT`` environment variable.

Managed: SLURM
--------------

Noether enters "managed" mode automatically whenever ``SLURM_PROCID`` is set in
the environment (i.e. the process was launched by ``srun``). In that case
Noether reads ``SLURM_NTASKS_PER_NODE``, ``SLURM_PROCID``, ``SLURM_JOB_NODELIST``,
and ``SLURM_JOB_ID`` to set up the process group — it does **not** spawn
worker processes itself.

This means SLURM must launch **one task per GPU**:

.. important::

   In managed mode you must set ``gpus_per_node``, and ``tasks_per_node`` must
   equal ``gpus_per_node`` (one rank per GPU). If you only set
   ``gpus_per_node`` via :doc:`noether-train-submit-job <launch_job>`,
   ``tasks_per_node`` is auto-derived to match. With a hand-written
   ``sbatch``/``srun`` script you have to set both yourself — Noether will
   refuse to start if the count of visible devices does not match
   ``SLURM_NTASKS_PER_NODE``.

Option A — custom ``sbatch`` script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this when you already have a cluster job template, or need SLURM
directives that aren't exposed through ``submitit``.

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=noether-train
   #SBATCH --partition=gpu
   #SBATCH --nodes=2
   #SBATCH --gpus-per-node=4
   #SBATCH --ntasks-per-node=4         # MUST equal --gpus-per-node
   #SBATCH --cpus-per-task=16
   #SBATCH --mem=256G
   #SBATCH --time=12:00:00
   #SBATCH --output=logs/%j.out

   source .venv/bin/activate

   srun uv run noether-train --hp configs/train_shapenet.yaml \
       +experiment/shapenet=upt \
       +accelerator=gpu \
       dataset_root=/data/shapenet_car

The combination ``nodes=2`` × ``gpus_per_node=4`` × ``tasks_per_node=4`` gives
``world_size=8``; ``trainer.effective_batch_size`` must be a multiple of 8.

Option B — ``noether-train-submit-job``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For configs that already declare a ``slurm:`` section, the submitter validates
the config on the login node and submits via ``submitit``. See
:doc:`launch_job` for the full reference.

.. code-block:: yaml

   # configs/train_shapenet.yaml
   slurm:
      folder: /home/%u/logs/shapenet
      slurm_partition: gpu
      nodes: 2
      gpus_per_node: 4        # tasks_per_node defaults to 4 (= gpus_per_node)
      cpus_per_task: 16
      mem_gb: 256
      timeout_min: 720
      slurm_setup:
         - source .venv/bin/activate

   trainer:
      effective_batch_size: 64   # multiple of world_size = nodes * gpus_per_node = 8
      ...

.. code-block:: bash

   uv run noether-train-submit-job --hp configs/train_shapenet.yaml \
       +experiment/shapenet=upt \
       dataset_root=/data/shapenet_car

If you set ``tasks_per_node`` explicitly in the ``slurm:`` section, it must
match ``gpus_per_node``; otherwise Noether will fail at startup when the
visible-device count does not match ``SLURM_NTASKS_PER_NODE``.
