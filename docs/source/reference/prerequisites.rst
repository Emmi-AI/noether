Prerequisites
=============

Welcome! Before you install the **Noether Framework**, let's make sure your system is ready. This checklist will help
make the installation process smooth and successful.


System Requirements
-------------------

Operating System
~~~~~~~~~~~~~~~~
The framework is primarily developed and tested on **Linux (Ubuntu 24.04+)** and **macOS (14.0+)**.

.. warning::
   Windows is not officially supported at this time. While it may be possible to run the framework, you may encounter
   issues, and we cannot provide support.

Python
~~~~~~
A 64-bit installation of **Python 3.10 or newer** is required. We recommend to use **3.12**. You can download it from
the official `Python website <https://www.python.org/>`_.

Required Tools
--------------

* **Git**: Used to clone the framework's source code repository. You can install it from
  `git-scm.com <https://git-scm.com/install/>`_.

* **uv**: Our recommended package manager for installing Python dependencies. Please install it by following
  the official `uv documentation <https://docs.astral.sh/uv/>`_.

Hardware Requirements
---------------------

* **CPU/RAM**: A modern multi-core CPU and at least **16 GB** of RAM are recommended for a good experience.

* **GPU (Highly Recommended)**: For training models, an **NVIDIA GPU** with CUDA support is essential. For reference,
  our internal cluster uses H100 GPUs, but it is possible to train models on less advanced GPUs.

.. important::
   **Check Your CUDA Version**

   Our framework depends on PyTorch, which requires a specific CUDA version.

   1.  Check your NVIDIA driver's CUDA capability by running this command in your terminal: ``nvidia-smi``
   2.  Note the CUDA Version shown in the top right.
   3.  Compare this to the PyTorch version specified in the ``pyproject.toml`` file(s) to ensure compatibility.
