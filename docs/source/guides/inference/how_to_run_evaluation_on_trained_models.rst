How to Run Evaluation on Trained Models
======================================

The evaluation pipeline allow you to run inference and compute metrics on a model that has been previously trained. This is useful for evaluating best checkpoints on test sets, generating visualizations, or performing late-stage analysis without starting a full training run.

Overview
--------

The evaluation process uses the ``noether-eval`` CLI command, which:
1. Loads the original training configuration from the run directory (provided as `input_dir`).
2. Merges it with an inference-specific configuration.
3. Initializes the model using weights from a specified checkpoint.
4. Executes the configured trainer's ``eval()`` method, which runs all configured callbacks.

The outputs are stored in a *new* directory separate from the training outputs, preserving the original run data.

The CLI: ``noether-eval``
-------------------------

The command is installed via the ``noether`` package. It utilizes Hydra for configuration management and supports merging multiple YAML files or command-line overrides.

Required Arguments
~~~~~~~~~~~~~~~~~~

``noether-eval`` is also using Hydra. Config can be provided via the ``--hp`` flag or directly as CLI arguments. The following argument is required:

- ``input_dir``: The absolute or relative path to the directory of the training run you wish to evaluate. This directory must contain a ``hp_resolved.yaml`` file.

Supported Callbacks
~~~~~~~~~~~~~~~~~~~
All callbacks that are compatible with evaluation mode can be used here. This includes metric computation callbacks, visualization callbacks, and any custom callbacks you may have implemented.

Not all callbacks are suitable for evaluation, for example the checkpoint saving callbacks are not relevant during evaluation and will be ignored.

Callbacks can check whether they are running in evaluation mode by checking ``interval_type == "eval"`` in the :py:meth:`~noether.core.callbacks.periodic.PeriodicCallback._periodic_callback` method. More details on how to implement custom callbacks can be found in :py:class:`noether.core.callbacks.periodic.PeriodicCallback` and :py:class:`noether.core.callbacks.periodic.PeriodicIteratorCallback`.

Examples
~~~~~~~~

Basic evaluation using the latest checkpoint. This runs the same callbacks as configured during training:

.. code-block:: bash

   noether-eval +input_dir=outputs/2026-01-10/10-00-00

Evaluation on a specific checkpoint:

.. code-block:: bash

   noether-eval +input_dir=outputs/2026-01-10/10-00-00 resume_checkpoint=best_accuracy \
   --hp configs/inference/visualization.yaml

Run evaluation with modified callbacks, for example to calculate offline losses on the test set:

.. code-block:: yaml

   # configs/inference/custom_eval_callbacks.yaml
   trainer:
   callbacks:
   - kind: noether.training.callbacks.OfflineLossCallback
      dataset_key: test
      name: OfflineLossCallback

.. code-block:: bash

    noether-eval +input_dir=outputs/2026-01-10/10-00-00 --hp configs/inference/custom_eval_callbacks.yaml


Configuration Merging
---------------------

``noether-eval`` performs a deep merge of configurations:

1. **Base**: The stored config from the ``input_dir`` (looked up from``hp_resolved.yaml``)
2. **Override**: The configuration provided via ``--hp`` or direct CLI arguments.

This allows you to easily switch datasets, modify callback parameters, or change evaluation settings while keeping the model architecture and other training-time settings intact.

Inference Runner
----------------

The ``InferenceRunner`` (found in ``src/noether/inference/runners/inference_runner.py``) is responsible for setting up the environment similarly to the training runner but with key differences:

- **Weight Loading**: It uses the ``PreviousRunInitializer``, which only loads model weights and skips optimizer/scheduler states.
- **Eval Mode**: It calls ``trainer.eval(model)`` instead of ``trainer.train(model)``.

Trainer Evaluation Mode
-----------------------

All evaluation mode is doing, is executing the configured callbacks on the saved model weights.

This way we can reuse the same callback implementations for both training and evaluation, ensuring consistency in metrics computation and visualization generation.

By default the same callbacks used during training are also used during evaluation. 
However, you can customize this by modifying the configuration passed to ``noether-eval`` to include different or additional callbacks as needed.
This way you can, for example, add extra visualization callbacks or change metric logging behavior specifically for evaluation runs.
