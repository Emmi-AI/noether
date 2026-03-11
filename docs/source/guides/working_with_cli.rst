Working with the CLI
====================

The **Noether CLI** provides a convenient way of performing high-level actions.

Some of the Python modules exposes unique functionality and therefore its own set of commands.

A global ``--debug`` flag enables extended debug logging, useful for narrowing down issues, for example:

.. code-block:: bash

   noether-data --debug aws estimate ...

In this documentation we will be using CLI for ``noether.io`` module.

.. important::

    Whenever you use a CLI from the **Noether** framework - your data stays on your local machine!

    Emmi AI doesn't store or collect your data.

Credential Setup
----------------

To communicate with services, you must configure credentials:

1. Create a file under your profile root, e.g. ``/users/username/.config/emmi/config.json``.
2. Populate the file (replace empty strings with your values; unused services may stay empty):

.. code-block:: json

    {
        "huggingface": {
            "HF_TOKEN": ""
        },
        "aws": {
            "AWS_ACCESS_KEY_ID": "",
            "AWS_SECRET_ACCESS_KEY": "",
            "AWS_SESSION_TOKEN": "",
            "AWS_REGION": "",
            "AWS_DEFAULT_REGION": ""
        }
    }

.. note::

   For AWS the values ``AWS_SESSION_TOKEN``, ``AWS_REGION``, and ``AWS_DEFAULT_REGION`` are optional.
   In case when ``AWS_ACCESS_KEY_ID`` starts with ``ASIA...`` the ``AWS_SESSION_TOKEN`` **needs** to be provided.

Testing the Credentials
-----------------------

Verify your setup by running the ``estimate`` command, which fetches metadata and reports the estimated size:

**Hugging Face:**

.. code-block:: bash

    noether-data huggingface estimate EmmiAI/AB-UPT

**AWS:**

.. code-block:: bash

    noether-data aws estimate noaa-goes16 ABI-L1b-RadC/2023/001/00/

If you see no errors — congratulations, your setup works!

Scaffolding a New Project
-------------------------

The ``noether-init`` command generates a complete Noether training project with all required modules and configurations.

.. code-block:: bash

   noether-init my_project \
       --model upt \
       --dataset shapenet_car \
       --dataset-path /path/to/shapenet_car

**Required arguments:**

- ``project_name`` (positional) — project name, must be a valid Python identifier (no hyphens)
- ``--model, -m`` — model architecture (``transformer``, ``upt``, ``ab_upt``, ``transolver``)
- ``--dataset, -d`` — dataset (``shapenet_car``, ``drivaernet``, ``drivaerml``, ``ahmedml``, ``emmi_wing``)
- ``--dataset-path`` — path to dataset on disk

**Optional arguments:**

- ``--optimizer, -o`` — optimizer, default: ``adamw`` (also: ``lion``)
- ``--tracker, -t`` — experiment tracker, default: ``disabled`` (also: ``wandb``, ``trackio``, ``tensorboard``)
- ``--hardware`` — hardware target, default: ``gpu`` (also: ``mps``, ``cpu``)
- ``--project-dir, -l`` — parent directory for the project folder, default: current directory
- ``--wandb-entity`` — W&B entity name (only used with ``--tracker wandb``)

For a detailed walkthrough and the generated project structure, see :doc:`/tutorials/scaffolding_a_new_project`.
