Training First Model (with Configs)
===================================

Prerequisites
--------------

- You have cloned the **Noether Framework**
- You have a ``tutorial/`` folder in the repo root

In this example we look at ``ShapeNet-Car`` dataset.

The fetching and preprocessing instructions are in the ``README.md`` located in the
``src/noether/data/datasets/cfd/shapenet_car/`` folder. Review them first and proceed with the next step when ready.

Execution
---------
Now that we have the dataset ready, we can start training our first model using the provided configuration files.
Note by default the outputs (checkpoints, logs, etc.) will be stored in the ``outputs/`` folder in the repo root.

.. code-block:: bash

    uv run noether-train --hp tutorial/configs/train_shapenet.yaml \
        +experiment/shapenet=upt \
        tracker=disabled \
        dataset_root=/Users/user/data/shapenet_car

We expose ``noether-train`` command via `uv` for your convenience to run the training. Here we specify the location
of the training configuration ``train_shapenet.yaml``, experiment type ``shapenet=upt`` (it means that we run  ``UPT``
model from the ``experiment/shapenet/`` folder, and the dataset root.

Additionally, we can alter the hyperparameters right from the CLI:

.. code-block:: bash

    uv run noether-train --hp tutorial/configs/train_shapenet.yaml \
        +experiment/shapenet=upt \
        dataset_root=/Users/user/data/shapenet_car \
        trainer.precision=fp32 \
        +accelerator=mps \
        tracker=disabled \
        +seed=1

Here we enforce ``fp32`` precision for training, ``mps`` (Apple Silicon) as our accelerator, ``tracker=disabled`` means
that there will be no experiment tracking used. Details on how to configure experiment tracking can be found in :doc:`guides/training/experiment_tracking`.

Note the specific syntax that comes from `Hydra <https://hydra.cc/>`_:

- ``key=value`` → override an existing config key.
- ``+key=value`` → add a new key that isn’t in the config schema yet (``Hydra`` calls this “force-add”).

And for config groups:

- ``+experiment/shapenet=upt`` → select a config from a config group (compose ``experiment/shapenet/upt.yaml``).

The ``+`` here commonly means “this group wasn’t already set in the base config, so add it”.

Once it's running, you should observe the logs appearing in your terminal with the status updates from your first
training using the **Noether Framework**!
