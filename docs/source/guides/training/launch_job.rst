How to launch a SLURM job from the command line
===============================================

The ``noether-train-submit-job`` command validates a Hydra training config and
submits it to SLURM via `submitit <https://github.com/facebookincubator/submitit>`_.
You pass the config path, optionally with Hydra-style overrides; the script
validates the result against your schema before any job is sent.

Single-job submission
---------------------

.. code-block:: bash

   noether-train-submit-job /path/to/recipes/aero_cfd/configs/train_shapenet.yaml \
       +experiment/shapenet=transformer \
       +seed=1 \
       tracker=disabled \
       dataset_root=/path/to/datasets/shapenet_car/

Sweeping with ``--multirun`` (SLURM array jobs)
-----------------------------------------------

Add ``--multirun`` (or ``-m``) and use Hydra's comma sweep syntax to launch a
single SLURM array job whose tasks span the cross-product of overrides. Every
combination is validated upfront on the login node — submission is aborted on
the first invalid combo, before SLURM sees anything.

.. code-block:: bash

   noether-train-submit-job --hp configs/train_shapenet.yaml -m \
       +seed=1,2,3 trainer.lr=1e-3,1e-4

The example above produces a 6-task SLURM array. Cap the concurrency with
``slurm.slurm_array_parallelism``.

Sweeping over fields under ``slurm:`` is **not** supported — the SLURM
allocation must be identical across all array tasks. The submitter rejects
configs whose ``slurm`` section differs between sweep combinations.

Previewing without submitting
-----------------------------

.. code-block:: bash

   noether-train-submit-job --hp configs/train_shapenet.yaml --dry-run

Prints the resolved submitit parameters and per-task command(s) without
contacting SLURM.

The ``slurm`` section of the config
-----------------------------------

The ``slurm`` section is required. Field names mirror the keyword arguments
accepted by :meth:`submitit.AutoExecutor.update_parameters`. An example:

.. code-block:: yaml

   slurm:
      folder: /home/%u/logs/shapenet_car   # %u = username; also default output_path
      name: shapenet_experiment
      slurm_partition: compute
      nodes: 1
      tasks_per_node: 1
      cpus_per_task: 28
      gpus_per_node: 1
      mem_gb: 64                              # gigabytes (float)
      timeout_min: 720                        # wall-clock minutes (int)
      slurm_setup:                            # commands run before the main task
         - source .venv/bin/activate
      slurm_additional_parameters:            # escape hatch for any sbatch flag
         nice: 0

Common fields:

- ``folder`` — directory where submitit writes the job script, pickled task,
  and stdout/stderr logs. Supports ``%u`` (current username) interpolation.
  Also serves as the default ``output_path`` for training runs when
  ``output_path`` is not set in the config. SLURM job-time patterns
  (``%j``, ``%A``, etc.) are not supported.
- ``name`` — job name (``--job-name``).
- ``slurm_partition`` — partition to submit to.
- ``nodes`` / ``tasks_per_node`` / ``cpus_per_task`` / ``gpus_per_node`` —
  resource allocation per the obvious ``sbatch`` flags. ``gpus_per_node``
  accepts ``"a100:4"``-style specs.
- ``mem_gb`` — memory per node in **gigabytes** (float).
- ``timeout_min`` — wall-clock limit in **minutes** (int).
- ``slurm_array_parallelism`` — cap concurrent tasks for ``--multirun`` arrays.
- ``slurm_setup`` — list of shell commands run inside the job before the main
  command (replaces the previous ``env_path`` field).
- ``slurm_additional_parameters`` — dict for any other sbatch directive
  (``nice``, ``reservation``, ``chdir``, ``account``, ``constraint``, ...).
  Keys are passed as ``--key=value`` to ``sbatch``.

.. important::

   **Job log files are owned by submitit.** The ``--output`` and ``--error``
   sbatch directives are intentionally *not* exposed as first-class fields.
   Submitit writes the job's stdout/stderr to:

   - ``<slurm.folder>/<job_id>_log.out``
   - ``<slurm.folder>/<job_id>_log.err``

   For array jobs, ``<job_id>`` becomes ``<array_master_id>_<task_idx>`` (e.g.
   ``42_0_log.out``). To direct logs to a per-experiment location, point
   ``slurm.folder`` at it. If you absolutely must override SLURM's output
   paths yourself, pass ``output``/``error`` via ``slurm_additional_parameters``
   — but be aware this disables submitit's log-tailing helpers such as
   ``job.stdout()`` and ``job.stderr()``.
