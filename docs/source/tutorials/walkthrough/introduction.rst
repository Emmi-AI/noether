Introduction
============

Welcome to the Noether Framework walkthrough!

This walkthrough demonstrates how to use the Noether Framework through a practical project based on the
experiments from Section 4.4 of the `AB-UPT paper <https://arxiv.org/pdf/2502.09692>`_.
While this walkthrough covers the core functionality of the framework, it does not cover every possible
feature or use case.


File structure of the project
-----------------------------

The walkthrough project follows this directory structure:

.. code-block:: text

   recipes/aero_cfd/
   ├── callbacks/      # Callbacks for evaluation, logging, and monitoring during training
   ├── configs/          # YAML files for configuring experiments using Hydra
   │   ├── callbacks/
   │   ├── data_specs/
   │   ├── dataset_normalizers/
   │   ├── datasets/
   │   ├── experiment/
   │   ├── model/
   │   ├── optimizer/
   │   ├── pipeline/
   │   ├── slurm/
   │   ├── tracker/
   │   ├── trainer/
   │   ├── train_ahmedml.yaml
   │   ├── train_caeml.yaml
   │   ├── train_drivaerml.yaml
   │   ├── train_drivaernet.yaml
   │   ├── train_shapenet.yaml
   │   └── train_wing.yaml
   ├── jobs/             # SLURM job scripts for running experiments on clusters
   ├── model/            # Model architecture definitions
   ├── pipeline/         # Data processing and collation pipeline
   └── trainers/         # Trainer classes that manage the training loop

**Minimal required structure** for any Noether project:

.. code-block:: text

   └── callbacks/        # Can be empty if only using default callbacks
   └── configs/          # Required: defines all configurations
   └── datasets/         # Required only for custom datasets
   └── pipeline/         # Required: defines data processing
   └── model/            # Required: defines model architectures
   └── trainers/         # Required: defines training logic

The ``configs/`` directory roughly mirrors the root folder structure; for each module or class
defined in the project, there is a corresponding configuration file.

.. tip::

   You can also scaffold this structure automatically using ``noether-init``.
   See :doc:`/tutorials/scaffolding_a_new_project` for details.


Core components
~~~~~~~~~~~~~~~

Every Noether project consists of the following core modules (in alphabetical order):

#. **Callbacks**: Classes that compute metrics and statistics at specific points during training. Can be empty when using only the framework's default callbacks.
#. **Configs**: YAML configuration files that define all hyperparameters, paths, and settings for the training pipeline.
#. **Dataset**: Provides the interface between raw data on disk and the multi-stage pipeline. Defines how to load individual tensors for each data sample. This walkthrough uses pre-implemented datasets, but you can create custom ones.
#. **Model**: Defines the model architecture and its forward pass.
#. **Pipeline**: Defines the multi-stage data pipeline that loads, processes, and collates individual samples into batches for training.
#. **Schemas**: Pydantic schemas that define the input data of each class we define in our project.
#. **Trainer**: The trainer loop takes batches from the pipeline, runs the model's forward pass and computes the loss.


Project setup
-------------

Clone the repository and set up the environment:

.. code-block:: bash

   git clone https://github.com/Emmi-AI/noether.git
   cd noether/
   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install emmiai-noether


For detailed installation and verification instructions, see
:doc:`/tutorials/getting_started_install_and_verify`.


Scaffolding a new project
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``noether-init`` CLI tool scaffolds a minimal, ready-to-train Noether project with all the
required components (callbacks, configs, datasets, models, pipelines, schemas, and trainer).
The generated project structure mirrors the template files in
``src/noether/scaffold/template_files/``.

To scaffold a new project:

.. code-block:: bash

   uv run noether-init my_project

This creates a project directory with a working training setup. To run it:

.. code-block:: bash

   cd my_project
   uv run noether-train --hp my_project/configs/base_experiment.yaml +seed=1 +devices=\"0\" tracker=disabled

We recommend scaffolding a project alongside this walkthrough to see what minimal
implementations look like. See :doc:`/tutorials/scaffolding_a_new_project` for the full guide.

.. important::

   All training commands must be run from inside the respective project or recipe folder
   (e.g., ``recipes/aero_cfd/``).

.. important::

   The Noether Framework by default runs on GPU. If no GPU is available, please add either
   ``+accelerator=cpu`` or ``+accelerator=mps`` to the command.
