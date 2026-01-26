Advanced Linux & NVIDIA Setup
=============================

Due to third-party dependencies like ``torch_geometric`` and ``torch_scatter``, extra steps are required to support
the newest NVIDIA architectures (e.g., Blackwell) on Linux.

Prerequisites
-------------

Ensure you have the relevant CUDA toolkit installed. The following example uses CUDA 12.9 to ensure NVIDIA Blackwell
architecture support.

1. **Install CUDA Toolkit:**

   .. code-block:: bash

        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt update
        sudo apt install -y cuda-toolkit-12-9

2. **Symlink the Installation:**

   Create a symlink for your fresh CUDA toolkit installation to a generic path (ensure you link your target version).

   .. code-block:: bash

        sudo ln -s /usr/local/cuda-12.9 /usr/local/cuda

3. **Configure Environment Variables:**

   Set these variables to help ``uv`` build third-party packages with the newest CUDA.

   .. code-block:: bash

        export CUDA_HOME=/usr/local/cuda
        export PATH=$CUDA_HOME/bin:$PATH
        export TORCH_CUDA_ARCH_LIST="12.0"
        export FORCE_CUDA=1

Validation
----------

Verify your compiler installation:

.. code-block:: bash

    which nvcc
    # Output: /usr/local/cuda/bin/nvcc

    nvcc --version
    # Should show: Cuda compilation tools, release 12.9...

Installing Custom Dependencies
------------------------------

By running ``uv sync`` you will be able to fetch all relevant dependencies and build the missing ones.

In case you need to reinstall specific components:

.. code-block:: bash

    uv pip install --no-binary torch-cluster torch-cluster --force-reinstall