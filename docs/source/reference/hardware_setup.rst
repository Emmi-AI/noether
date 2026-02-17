Hardware Allocation & Training Configurations
=============================================

**Noether** is designed to run seamlessly across diverse hardware environments — from local Apple Silicon (MPS) laptops
to large scale clusters. This guide details how to configure the training CLI for your specific setup.

Local Development (CPU & Apple Silicon)
---------------------------------------

We often use MacBooks for development and debugging. While slower than dedicated GPUs, the ``mps``
(Metal Performance Shaders) accelerator allows for decent iteration speeds on M-series chips.

**Key Configuration:**

- ``+accelerator=mps`` (or ``cpu``)
- ``trainer.precision=fp32`` (MPS has limited support for mixed precision)

**Example Command:**

.. code-block:: bash

   uv run noether-train --hp tutorial/configs/train_shapenet.yaml \
       +experiment/shapenet=upt \
       dataset_root=/Users/user/data/shapenet_car \
       trainer.precision=fp32 \
       +accelerator=mps \
       tracker=disabled \
       +seed=1


Shared Clusters (SLURM)
-----------------------

In managed environments (like university or corporate clusters), SLURM handles the resource allocation. **Noether**
detects the available resources automatically.

**Key Configuration:**

* ``+accelerator=gpu``
* No ``devices`` flag needed (by default will use all GPUs allocated by SLURM).

**Example Command:**

.. code-block:: bash

   # Assuming srun or sbatch has already allocated GPUs
   uv run noether-train --hp tutorial/configs/train_shapenet.yaml \
       +experiment/shapenet=upt \
       dataset_root=/data/shapenet_car \
       trainer.precision=bf16 \
       +accelerator=gpu \
       tracker=disabled


Non-Managed Workstations
------------------------

If you are running on a personal GPU workstation (e.g., a devbox with 4x GPUs) without a scheduler, you can manually
control which GPUs are visible to the training job using the ``devices`` flag.

**Key Configuration:**

- ``+devices="0"`` (Use only GPU 0)
- ``+devices="0,1"`` (Use GPU 0 and 1)

**Example Command:**

.. code-block:: bash

   uv run noether-train --hp tutorial/configs/train_shapenet.yaml \
       +experiment/shapenet=upt \
       dataset_root=/home/user/data/shapenet_car \
       +accelerator=gpu \
       +devices=\"0\" \
       tracker=disabled

.. note::
    **Note on escaping:** When passing list-like strings in Hydra/CLI, you often need to escape the quotes (e.g.,
    ``\"0,1\"``) to prevent your shell from interpreting them.