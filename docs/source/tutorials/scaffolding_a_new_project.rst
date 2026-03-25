Scaffolding a New Project
=========================

The ``noether-init`` command scaffolds a minimal, ready-to-train Noether project. It creates all required Python modules, Hydra configuration files, schemas, trainers,
and callbacks, giving you a working starting point that you can adapt to your own use case.


Example Usage
-------------

.. code-block:: bash

   uv run noether-init my_project

This creates a ``my_project/`` directory. After completion,
``noether-init`` prints a summary of the configuration and the ``noether-train`` command to start training.

Arguments
---------

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Option
     - Values
     - Default
   * - ``project_name`` *(required)*
     - Positional argument. Must be a valid Python identifier (no hyphens).
     -
   * - ``--tracker, -t``
     - ``wandb``, ``trackio``, ``tensorboard``, ``disabled``
     - ``disabled``
   * - ``--hardware``
     - ``gpu``, ``mps``, ``cpu``
     - ``gpu``
   * - ``--project-dir, -d``
     - Parent directory for the project folder
     - current directory
   * - ``--wandb-entity``
     - W&B entity name (only with ``--tracker wandb``)
     - your W&B username

Generated Project Structure
---------------------------

The generated project contains:

.. code-block:: text

   my_project/
   ├── pyproject.toml            # Project config with emmiai-noether dependency
   └── my_project/               # Python package
       ├── __init__.py
       ├── callbacks/             # Training callbacks
       ├── configs/
       │   ├── tracker/           # Tracker configs (wandb, disabled, etc.)
       │   └── base_experiment.yaml  # Main training config
       ├── datasets/              # Dataset implementation
       ├── models/                # Model implementation
       ├── pipelines/             # Data pipeline
       ├── schemas/               # Configuration dataclasses
       └── trainer/               # Training loop implementation

Running Training
----------------

After scaffolding, start training with:

.. code-block:: bash

   cd my_project
   uv run noether-train --hp my_project/configs/base_experiment.yaml
