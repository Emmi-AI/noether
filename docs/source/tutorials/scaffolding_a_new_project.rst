Scaffolding a New Project
=========================

The ``noether-init`` command generates a complete, ready-to-train Noether project for
models and datasets supported out of the box by the framework. It creates all required Python modules, Hydra configuration
files, schemas, data pipelines, trainers, and callbacks, giving you a working starting point that you
can adapt to your own use case.

Prerequisites
-------------

Before scaffolding, download and preprocess the dataset you want to use. Each dataset has its own
fetching and preprocessing instructions — see the
`Dataset Zoo README <https://github.com/Emmi-AI/noether/blob/main/src/noether/data/datasets/README.md>`_
for an overview and links to dataset-specific guides.

Example Usage
-------------

.. code-block:: bash

   noether-init my_project \
       --model upt \
       --dataset shapenet_car \
       --dataset-path /path/to/shapenet_car

This creates a ``my_project/`` directory in the current working directory with a UPT model and the ``shapenet_car`` dataset.
After completion, ``noether-init`` prints a summary of the configuration and the corresponding
``noether-train`` command to start training.

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
   * - ``--model, -m`` *(required)*
     - ``transformer``, ``upt``, ``ab_upt``, ``transolver``
     -
   * - ``--dataset, -d`` *(required)*
     - ``shapenet_car``, ``drivaernet``, ``drivaerml``, ``ahmedml``, ``emmi_wing``
     -
   * - ``--dataset-path`` *(required)*
     - Path to the dataset on disk
     -
   * - ``--optimizer, -o``
     - ``adamw``, ``lion``
     - ``adamw``
   * - ``--tracker, -t``
     - ``wandb``, ``trackio``, ``tensorboard``, ``disabled``
     - ``disabled``
   * - ``--hardware``
     - ``gpu``, ``mps``, ``cpu``
     - ``gpu``
   * - ``--project-dir, -l``
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
   ├── configs/
   │   ├── callbacks/          # Training callback configs
   │   ├── data_specs/         # Data specification configs
   │   ├── dataset_normalizers/
   │   ├── dataset_statistics/
   │   ├── datasets/           # Dataset configs
   │   ├── experiment/         # Experiment configs (one per model)
   │   ├── model/              # Model architecture config
   │   ├── optimizer/          # Optimizer config
   │   ├── pipeline/           # Data pipeline config
   │   ├── tracker/            # Experiment tracker config
   │   ├── trainer/            # Trainer config
   │   └── train.yaml          # Main training config
   ├── model/                  # Model implementation
   ├── schemas/                # Configuration dataclasses
   ├── pipeline/               # Data processing (collators, sample processors)
   ├── trainers/               # Training loop implementation
   └── callbacks/              # Training callbacks

All Python files are wired up with correct imports for your chosen model, and all Hydra configs reference
your dataset path, optimizer, and tracker selections.

Running Training
----------------

After scaffolding, start training with:

.. code-block:: bash

   uv run noether-train \
       --config-dir my_project/configs \
       --config-name train \
       +experiment=upt
