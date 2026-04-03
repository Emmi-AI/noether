External Aerodynamics (Python)
==============================

This recipe is the Python-configuration counterpart of the
:doc:`/tutorials/walkthrough/index` recipe (``recipes/aero_cfd/``).
While the original recipe uses YAML configuration files, this version
defines the same experiments entirely in Python using the preset-based
interface.

Source code: `recipes/aero_cfd_python/ <https://github.com/Emmi-AI/noether/tree/main/recipes/aero_cfd_python>`_

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

   uv run python -m aero_cfd_python.train_emmi_wing

Before running, update the ``DATASET_ROOT`` and ``OUTPUT_PATH`` variables in the
script to point to your local data and output directories.

To train a different model architecture, edit the ``if __name__ == "__main__"``
block at the bottom of the script to call the desired function (e.g.,
``train_upt()`` instead of ``train_abupt()``).
