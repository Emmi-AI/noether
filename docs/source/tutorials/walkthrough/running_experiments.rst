Running the Experiments
=======================

For a general guide on launching training jobs, see :doc:`/guides/training/launch_job`.

.. important::

   All commands in this section must be run from inside the recipe folder (``recipes/aero_cfd/``).


Running SLURM jobs
------------------

The Noether Framework provides the ``noether-train-submit-job`` CLI for submitting SLURM jobs.
It reads SLURM parameters from a ``slurm`` config group in your experiment configuration.
This recipe defines its SLURM defaults in :source:`configs/slurm/slurm_config.yaml <../../../../recipes/aero_cfd/configs/slurm/slurm_config.yaml>`:

.. literalinclude:: ../../../../recipes/aero_cfd/configs/slurm/slurm_config.yaml
   :language: yaml

For a detailed guide on configuring and using ``noether-train-submit-job``, see
:doc:`/guides/training/launch_job`.

Alternatively, this recipe includes hand-written job scripts. To run all the models for
ShapeNet-Car:

.. code-block:: bash

   sbatch jobs/train_shapenet.job

The same applies to :source:`jobs/train_ahmedml.job <../../../../recipes/aero_cfd/jobs/train_ahmedml.job>` and :source:`jobs/train_drivaerml.job <../../../../recipes/aero_cfd/jobs/train_drivaerml.job>`, which can be found in the
``jobs/`` directory.

We also provide the config files to run the experiments for
`DrivAerNet++ <https://arxiv.org/abs/2406.09624>`_
(:source:`train_drivaernet.yaml <../../../../recipes/aero_cfd/configs/train_drivaernet.yaml>`) and the
`Emmi-Wing <http://arxiv.org/html/2511.21474v1>`_
(:source:`train_wing.yaml <../../../../recipes/aero_cfd/configs/train_wing.yaml>`), however, those experiments are not part of this walkthrough.

.. warning::

   This assumes you have access to a SLURM-based system. If not, please review the job files
   to see the commands used to run the experiments.

**Job arrays:**

In the ``jobs/experiments/`` folder, we define job arrays (i.e., arrays with different
experiments/jobs) for all the experiments we want to run. You can add extra rows with different
seeds or experiment variants to these ``*.txt`` files as needed.

The flag ``#SBATCH --array=...`` defines how to run the job array:

- ``#SBATCH --array=1-10``: Runs rows 1 to 10 from ``./jobs/experiments/shapenet_experiments.txt``
- ``#SBATCH --array=1,5,9``: Runs rows 1, 5, and 9
- ``#SBATCH --array=1-10%5``: Runs rows 1 to 10, but with a maximum of 5 jobs running
  simultaneously. When one of the 5 jobs finishes, the next job in the array will start. This is
  especially useful for large job arrays when you don't want to occupy the entire cluster.


Running a single experiment
---------------------------

To run a single experiment, execute the following command:

.. code-block:: bash

   uv run noether-train \
     --hp configs/train_shapenet.yaml \
     +experiment/shapenet=transformer tracker=disabled +seed=1

.. important::

   Please set the ``dataset_root`` in either the config files or via the command line override.


Running multi-GPU experiments
-----------------------------

When running outside of SLURM, use ``uv run noether-train`` as shown above. This will spawn one
process for every GPU that is available on the system and visible via ``CUDA_VISIBLE_DEVICES``.
You can also configure the devices by adding ``devices="0,1,2,4"``, for example, to the root
config.

.. important::

   If you train on more than 1 GPU, ensure that ``effective_batch_size`` is at least equal to
   the number of GPUs used. Multi-node training is currently not supported.

Example of a multi-GPU SLURM job:

.. code-block:: bash

   srun --nodes=1 --partition=compute --gpus-per-node=2 --mem=64GB \
     --ntasks-per-node=2 --kill-on-bad-exit=1 --cpus-per-task=28 \
     uv run noether-train \
       --hp configs/train_shapenet.yaml \
       +experiment/shapenet=transformer tracker=disabled \
       trainer.effective_batch_size=2


Running inference and evaluation on a trained model
---------------------------------------------------

Once a training run finishes, you can re-run its callbacks against any saved
checkpoint with ``noether-eval``. Point it at the run output directory (the
folder that contains ``hp_resolved.yaml``):

.. code-block:: bash

   uv run noether-eval run_dir=outputs/<run_id>/train

That single argument is enough — ``noether-eval`` reads the original training
config from ``hp_resolved.yaml``, restores the latest checkpoint, and re-runs
the configured callbacks. Whether that yields metric numbers, saved
predictions, or visualizations depends on which callbacks were configured —
the runner is a thin post-training callback executor and doesn't care.

Common overrides:

.. code-block:: bash

   # Use the best checkpoint instead of the latest. The tag is
   # `best_model.<metric>` (slashes flattened to dots); the metric comes from
   # the run's BestCheckpointCallback config — e.g. `loss/test/total`:
   uv run noether-eval run_dir=outputs/<run_id>/train resume_checkpoint=best_model.loss.test.total

   # Disable experiment tracking for a one-off run
   uv run noether-eval run_dir=outputs/<run_id>/train tracker=disabled

Any training-time config key (``trainer.*``, ``tracker.*``, ``model.*``, etc.)
can be overridden the same way — no Hydra ``+`` prefix needed. To plug in
extra callbacks for evaluation, prediction saving, or visualization, drop
them into a small YAML and pass it via ``--hp``:

.. code-block:: bash

   uv run noether-eval run_dir=outputs/<run_id>/train --hp configs/eval_extra.yaml

For the full reference (custom output directories, hardware overrides,
prediction-saving callback examples), see
:doc:`/guides/inference/how_to_run_evaluation_on_trained_models`.


Resuming training after interruption
-------------------------------------

To resume training after an error or interruption, simply add ``resume_run_id: <RUN_ID>``
(and ``resume_stage_name`` if a ``stage_name`` was used in the previous run) to the training
configuration (either in the YAML file or via the CLI). Training will continue from the last
saved epoch checkpoint.

**Example:**

.. code-block:: bash

   uv run noether-train \
     --hp configs/train_shapenet.yaml \
     +experiment/shapenet=transformer \
     resume_run_id=<run_id> resume_stage_name=<stage_name>

Optionally, you can change the ``stage_name`` to make it clear that checkpoints stored for this
run are from a continued training run.


Initializing model weights
--------------------------

To initialize a model with weights from a previous training run, add an initializer
configuration to the model config:

.. code-block:: yaml

   model:
     # ... model configuration
     initializers:
       - kind: noether.core.initializers.PreviousRunInitializer
         run_id: <run_id>
         model_name: ab_upt
         checkpoint_tag: latest  # Options: 'latest', 'best', or specific checkpoint

**Required parameters:**

- ``run_id``: The run identifier from the previous training run
- ``model_name``: The name of the model to load weights from
- ``checkpoint_tag``: Which checkpoint to use (``latest``, ``best``, or a specific epoch number)

**Optional parameters:**

- ``model_info``: Additional checkpoint metadata (e.g., ``ema=0.9999`` for exponential moving
  average weights, or specific loss metric identifiers for best checkpoints). Leave empty for
  standard checkpoints.


WandB tracker
-------------

We implemented a Weights and Biases (WandB) tracker to log during training and evaluation:

.. literalinclude:: ../../../../recipes/aero_cfd/configs/tracker/development_tracker.yaml
   :language: yaml

Simply add your own WandB entity and project to start logging.

For more details on experiment tracking, see :doc:`/guides/training/experiment_tracking`.


Extra utilities and tips
------------------------

- **Output path**: The output path is undefined by default and must be configured. In this
  walkthrough, we set it to ``./outputs``. The Noether Framework will use the generated
  ``run_id`` to store the checkpoints for each training run in subfolders.
- **Physics features**: You can set ``physics_features`` to ``true`` for the multi-stage
  ``AeroMultistagePipeline``. This only works for ShapeNet-Car and will add the SDF and normal
  vectors to the coordinate inputs. However, we never properly utilized these features in our
  experiments, and they are not implemented for other datasets.
- **Code snapshots**: By default, a snapshot of the codebase is stored as part of the
  checkpoints for reproducibility.
- **Batch size considerations**: Almost all experiments we ran for the AB-UPT paper use a batch
  size of 1. However, the data loading pipeline is implemented to work with batches larger than
  1 (including with physics features). Note that we never thoroughly validated these results or
  checked for potential training/data loading instabilities with larger batch sizes.
- **Effective batch size and gradient accumulation**: The ``effective_batch_size`` parameter
  defines the total number of samples processed before performing an optimizer step (also known
  as the "global batch size"). In multi-GPU setups, the local batch size per device is calculated
  as ``effective_batch_size / number of GPUs``. When gradient accumulation is enabled, the batch
  size is further divided by the number of accumulation steps. To enable gradient accumulation,
  set the ``max_batch_size`` parameter. For example, with ``max_batch_size=2`` and
  ``effective_batch_size=8``, the framework will perform 4 gradient accumulation steps before
  updating the model weights.
