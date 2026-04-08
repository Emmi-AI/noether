External Aerodynamics (Python Scripts)
======================================

.. note::

   See the :doc:`/tutorials/walkthrough/index` for the full tutorial.

Source code: `recipes/aero_cfd/scripts/ <https://github.com/Emmi-AI/noether/tree/main/recipes/aero_cfd/scripts>`_

Available training scripts
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Script
     - Dataset
     - Source
   * - ``train_ahmedml.py``
     - AhmedML (CAEML benchmark)
     - `CAEML <https://caeml.org/>`_
   * - ``train_drivaerml.py``
     - DrivAerML (CAEML benchmark)
     - `CAEML <https://caeml.org/>`_
   * - ``train_drivaernet.py``
     - DrivAerNet++
     - `DrivAerNet <https://github.com/Mohamedelrefaie/DrivAerNet>`_
   * - ``train_emmi_wing.py``
     - Emmi Wing
     - `EmmiAI /Emmi Wing <https://github.com/Emmi-AI/Emmi-Wing/>`_
   * - ``train_shapenet_car.py``
     - ShapeNet Car
     - `ShapeNet <https://shapenet.org/>`_

Each script contains functions for training different model architectures
(AB-UPT, UPT, Transformer, Transolver).

How to run
----------

From the ``recipes`` directory run:

.. code-block:: console

   uv run python -m aero_cfd.scripts.train_shapenet_car \
     --dataset-root /path/to/shapenet_car \
     --output-path /path/to/outputs \
     --accelerator gpu \
     --model abupt

Available arguments:

- ``--dataset-root`` **(required)** — Path to the dataset.
- ``--output-path`` **(required)** — Path to store training outputs.
- ``--accelerator`` — Accelerator to use: ``cpu``, ``gpu``, or ``mps`` (default: ``gpu``).
- ``--model`` — Model architecture: ``abupt``, ``upt``, ``transformer``, or ``transolver`` (default: ``abupt``).
