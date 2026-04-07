Configuration
=============

.. note::

   This section assumes familiarity with `Hydra <https://hydra.cc/docs/intro/>`_ configuration
   management and `Pydantic <https://docs.pydantic.dev/latest/>`_ schemas. If you're new to these
   tools, we recommend reviewing their official documentation before proceeding.

The configuration is the backbone of the Noether Framework, enabling reproducible, modular, and
type-safe experiment definitions. All experiments are defined through **YAML configuration files**
that use:

- **Hydra** for hierarchical composition and command-line overrides
- **Pydantic** for runtime data validation and type safety

For a higher-level overview of how configs and code-driven workflows compare, see
:doc:`/noether/key_concepts`.


Configuration architecture
--------------------------

This tutorial uses a **hierarchical configuration pattern** where:

#. **Base configurations** define default settings for each component (datasets, models, trainers, etc.)
#. **Experiment configurations** compose and override base configs for specific experiments
#. **Command-line overrides** allow quick parameter sweeps without file changes

The main entry point for any experiment is a top-level configuration file like
:source:`configs/train_shapenet.yaml <../../../../recipes/aero_cfd/configs/train_shapenet.yaml>`,
which serves as the composition root that brings together all required components.


Example: ShapeNet-Car configuration
------------------------------------

:source:`train_shapenet.yaml <../../../../recipes/aero_cfd/configs/train_shapenet.yaml>` demonstrates the structure of a complete experiment configuration.
Let's break down its key components:

.. literalinclude:: ../../../../recipes/aero_cfd/configs/train_shapenet.yaml
   :language: yaml
   :lines: 1-25

Each entry like ``dataset_normalizers: shapenet_dataset_normalizers`` tells Hydra to load
:source:`configs/dataset_normalizers/shapenet_dataset_normalizers.yaml <../../../../recipes/aero_cfd/configs/dataset_normalizers/shapenet_dataset_normalizers.yaml>`
and merge it into the final configuration.

The ``???`` marker indicates required fields that must be specified in experiment configs.
The ``_self_`` marker controls when the current file's values override inherited ones (placing
it last gives the current file the highest priority).

**Complete configuration structure:**

To run an experiment, you need configurations for:

#. **Model**: Architecture and hyperparameters
#. **Trainer**: Trainer config
#. **Callbacks**: :doc:`Evaluation, logging, and monitoring </guides/training/use_callbacks>`
#. **Tracker**: :doc:`tracker </guides/training/experiment_tracking>`
#. **Dataset(s)**: Dataset config
#. **Pipeline**: Data preprocessing and collation
#. **Optimizer**: Optimization algorithm

Most components remain constant across experiments on the same dataset.
For example, when training different models on ShapeNet-Car, only the ``model`` and ``tracker``
configurations typically change, while ``dataset``, ``pipeline``, ``trainer``, and ``callbacks``
remain fixed.

**Example: Dataset configuration**

The base dataset configuration
:source:`configs/datasets/shapenet_dataset.yaml <../../../../recipes/aero_cfd/configs/datasets/shapenet_dataset.yaml>`
demonstrates config composition:

.. literalinclude:: ../../../../recipes/aero_cfd/configs/datasets/shapenet_dataset.yaml
   :language: yaml

Notice the ``${variable_name}`` references? These resolve to values defined in the top-level
``train_shapenet.yaml``. This pattern avoids duplication: ``dataset_root`` is defined once, used
everywhere.

**Config groups and directory structure:**

The ``configs/`` directory roughly mirrors the component structure:

.. code-block:: text

   configs/
   ├── train_shapenet.yaml          # Top-level composition
   ├── datasets/                    # Dataset config group
   │   ├── shapenet_dataset.yaml
   │   ├── ahmedml_dataset.yaml
   │   └── ...
   ├── model/                       # Model config group
   │   ├── transformer.yaml
   │   ├── upt.yaml
   │   └── ...
   ├── trainer/                     # Trainer config group
   │   └── shapenet_trainer.yaml
   └── experiment/                  # Experiment-specific overrides
       └── shapenet/
           ├── transformer.yaml
           ├── upt.yaml
           └── ...


Defining experiment configurations
-----------------------------------

Experiment-specific configurations compose base configs and apply targeted overrides.
An experiment file should:

#. Select a specific model variant
#. Choose a :doc:`tracker <../../guides/training/experiment_tracking>` (W&B, trackio, Tensorboard or disabled)
#. Override any experiment-specific hyperparameters

**Example: Transformer experiment**

The Transformer experiment configuration
:source:`configs/experiment/shapenet/transformer.yaml <../../../../recipes/aero_cfd/configs/experiment/shapenet/transformer.yaml>`:

.. literalinclude:: ../../../../recipes/aero_cfd/configs/experiment/shapenet/transformer.yaml
   :language: yaml

**Breaking down the experiment config:**

- ``override /model: transformer``: Use :source:`configs/model/transformer.yaml <../../../../recipes/aero_cfd/configs/model/transformer.yaml>` instead of the
  placeholder ``???`` in the base config
- ``override /tracker: development_tracker``: Select the W&B tracker configuration
- ``override /optimizer: lion``: Override the default AdamW optimizer with Lion
- ``trainer.precision: float16``: Override the trainer's default ``float32`` precision

The ``override`` keyword ensures the experiment's choice takes precedence over any defaults,
preventing accidental config merging issues.

**Creating new experiments:**

To run a different model on the same dataset:

#. Create a new experiment file (e.g., ``configs/experiment/shapenet/my_model.yaml``)
#. Specify the model config to use
#. Add any model-specific overrides
#. Keep tracker and other settings as needed


Running experiments
-------------------

**Basic execution:**

To train a model with a specific configuration (from the ``recipes/aero_cfd/`` directory):

.. code-block:: bash

   uv run noether-train --hp configs/train_shapenet.yaml \
     +experiment/shapenet=transformer tracker=disabled trainer.max_epochs=10

.. code-block:: bash

   uv run noether-train --hp configs/train_shapenet.yaml \
     +experiment/shapenet=ab_upt tracker=disabled trainer.max_epochs=10

To enable experiment tracking, simply remove the ``tracker=disabled`` override:

.. code-block:: bash

   uv run noether-train --hp configs/train_shapenet.yaml \
     +experiment/shapenet=transformer

.. important::

   All training commands must be run from inside the recipe folder (``recipes/aero_cfd/``).

.. warning::

   Make sure to either set ``dataset_root`` in ``train_shapenet.yaml`` or add it to the
   command line via ``dataset_root="<path to dataset root>"``.

You'll need to configure your W&B API key on first run and update
:source:`configs/tracker/development_tracker.yaml <../../../../recipes/aero_cfd/configs/tracker/development_tracker.yaml>` with your project details.

**Single parameter overrides:**

.. code-block:: bash

   uv run noether-train --hp configs/train_shapenet.yaml \
     +experiment/shapenet=transformer \
     trainer.max_epochs=100

**Multiple parameter overrides:**

To modify multiple related parameters (e.g., changing Transformer dimensions):

.. code-block:: bash

   uv run noether-train --hp configs/train_shapenet.yaml \
     +experiment/shapenet=transformer \
     model.hidden_dim=256 \
     model.transformer_block_config.num_heads=4

Note: When changing ``hidden_dim``, ensure ``num_heads`` divides it evenly
(i.e., ``hidden_dim % num_heads == 0``).

For more details on CLI-based training, see
:doc:`/tutorials/training_first_model_with_configs`. To run experiments using
Python code instead of YAML configs, see
:doc:`/tutorials/training_first_model_with_code`.

For launching training jobs on a SLURM cluster, see
:doc:`/guides/training/launch_job`.


Pydantic schemas for type safety
--------------------------------

While Hydra handles configuration composition, **Pydantic schemas** provide runtime validation
and type safety. Every class in the Noether Framework has a corresponding Pydantic schema that
validates configuration: checks types, ranges, and constraints before training begins.

**Schema hierarchy:**

All schemas in the Noether Framework follow an inheritance pattern. For example, model schemas
inherit from ``ModelBaseConfig``:

.. literalinclude:: ../../../../src/noether/core/schemas/models/base.py
   :language: python
   :pyobject: ModelBaseConfig
   :end-before: @property
   :dedent:

The ``extra: "forbid"`` setting ensures that typos in YAML files are caught immediately,
preventing silent configuration errors.


Example: Transformer configuration schema
------------------------------------------

All models in Noether use schema composition and validation. The schema hierarchy for the
Transformer models looks like:

.. code-block:: text

   ModelBaseConfig (base for all models)
       └── TransformerConfig (Transformer-specific config)
             └── TransformerBlockConfig (component config)

**TransformerBlockConfig** defines individual block parameters:

.. literalinclude:: ../../../../src/noether/core/schemas/modules/blocks.py
   :pyobject: TransformerBlockConfig

**TransformerConfig** extends the block config:

.. literalinclude:: ../../../../src/noether/core/schemas/models/transformer.py
   :pyobject: TransformerConfig

**Multiple inheritance** means ``TransformerConfig`` inherits:

- Model management from ``ModelBaseConfig`` (optimizer, freezing, etc.)
- Block parameters from ``TransformerBlockConfig`` (attention, MLP, etc.)
- Adds Transformer model parameters (``depth``)
- Overrides defaults (sets ``mlp_expansion_factor = 4``)


From schema to YAML
--------------------

Understanding the schema tells you which YAML fields are required and optional. Here is the
full Transformer model config:

.. literalinclude:: ../../../../recipes/aero_cfd/configs/model/transformer.yaml
   :language: yaml


Configuration inheritance
-------------------------

UPT and AB-UPT models support automatic configuration injection from parent to submodules for
shared parameters between parent and submodules.

When you set ``hidden_dim``, ``num_heads``, or ``mlp_expansion_factor`` at the top level of a
UPT config (or just ``hidden_dim`` for AB-UPT), these values automatically propagate to
submodules unless explicitly overridden. This reduces redundancy and keeps consistency across
your model architecture.

For a deeper understanding of how configuration inheritance works, see
:doc:`/reference/config_inheritance`.

**Example - UPT configuration:**

.. literalinclude:: ../../../../recipes/aero_cfd/configs/model/upt.yaml
   :language: yaml


Configuration schemas
---------------------

In the Noether Framework, Pydantic schemas are used to validate configuration at runtime.
Each component (model, trainer, dataset, etc.) has a corresponding config class that inherits
from a base schema.

For example, the trainer config schema for this recipe is defined in
:source:`trainers/aerodynamics_cfd.py <../../../../recipes/aero_cfd/trainers/aerodynamics_cfd.py>`:

.. literalinclude:: ../../../../recipes/aero_cfd/trainers/aerodynamics_cfd.py
   :language: python
   :pyobject: AerodynamicsCfdTrainerConfig
   :dedent:

The ``kind`` field in most configs specifies the class path for instantiation. The Factory
pattern uses this to dynamically import and instantiate the correct class with the validated
configuration.
