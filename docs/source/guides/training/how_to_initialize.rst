How to continue Training from a Checkpoint
==========================================

Noether stores model checkpoints using a structured file naming convention:

.. code-block:: text

   {model_name}_{model_info}_cp={checkpoint}_model.th

This naming scheme consists of three main components:

**Model Name** (``model_name``)
  A unique identifier for your model (e.g., ``transformer``, ``upt``, ``ab-upt``). This must be configured 
  in your model config and is used to identify which model the checkpoint belongs to.

**Checkpoint Identifier** (``checkpoint``)
  Indicates when in training the model was saved. Common values include:
  
  - ``latest`` — the most recent periodic checkpoint
  - ``best.{metric_name}`` — the checkpoint with the best value for a specific metric (e.g., ``best.accuracy``)
  - ``E10_U100_S200`` — a specific training point (epoch 10, update 100, sample 200)

**Model Info** (``model_info``) *(optional)*
  Additional metadata to distinguish special checkpoint variants. For example:
  
  - ``ema_factor=0.9999`` — Exponential Moving Average (EMA) weights
  - If omitted, the filename becomes: ``{model_name}_cp={checkpoint}_model.th``


Checkpoint Storage Location
----------------------------

Checkpoints are stored in the following directory structure:

.. code-block:: text

   <output_dir>/<run_id>/<stage_name>/checkpoints/{model_name}_{model_info}_cp={checkpoint}_model.th

Where:

- ``output_dir`` — Your configured output directory
- ``run_id`` — The unique identifier for the training run
- ``stage_name`` — The name of the training stage (if using multi-stage training)


Resuming Training from a Previous Run
--------------------------------------

To resume training from a checkpoint (e.g., after a crash or interruption), add the following to your root configuration file:

.. code-block:: yaml

   resume_run_id: <id of the previous run>
   resume_stage_name: <stage name from the previous run>
   stage_name: continue_training  # Optional: defaults to resumed stage name if not specified
   resume_checkpoint: <checkpoint>  # Optional: specify a particular checkpoint to resume from

If no `resume_checkpoint` is specified, the training will resume from the latest checkpoint of the specified stage.
For the `resume_checkpoint`, you can specify a particular checkpoint by using either the epoch (e.g., ``E10``) the update (e.g., ``U100``) or the sample (e.g., ``S200``) to resume from.
Do not use the fully specified checkpoint filename here (e.g., ``E10_U100_S200``).

Initializing Model Weights from a Previous Run
-----------------------------------------------

To initialize only the model weights (without resuming the full training state), use the :class:`~noether.core.initializers.PreviousRunInitializer` 
in your model configuration:

.. code-block:: yaml

   model:
     name: my_model
     kind: path.to.MyModel
     initializers:
       - kind: noether.core.initializers.PreviousRunInitializer
         run_id: <previous run_id>
         model_name: <model_name>
         stage_name: <stage_name>     # Optional: leave empty if no stage
         checkpoint: <checkpoint>      # e.g., latest, best.accuracy, E10_U100_S200
         model_info: <model_info>      # Optional: e.g., ema_factor=0.9999

This approach is useful for:

- Transfer learning from a pretrained model
- Fine-tuning on a new dataset
- Starting a new training run with pretrained weights


Use Cases
---------

**Full Training Resumption**
  Use ``resume_run_id`` and ``resume_stage_name`` when you need to continue training exactly where it left off, 
  preserving optimizer state, callback states (if any), and training/trainer progress/state.

**Weight Initialization Only**
  Use :class:`~noether.core.initializers.PreviousRunInitializer` when you want to start a training from scratch with 
  pretrained weights.