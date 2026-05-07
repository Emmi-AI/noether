How to Run Inference and Evaluation on Trained Models
=====================================================

``noether-eval`` is the post-training callback runner: point it at a finished
training run, pick a checkpoint, and it re-executes the configured callbacks
against those weights. Whether that means *evaluation* (computing metrics on
a held-out split) or *inference* (writing predictions to disk for downstream
analysis) depends entirely on which callbacks are configured — the runner
itself doesn't care, and most projects use it for both.

What this is good for
~~~~~~~~~~~~~~~~~~~~~

- **Evaluating on a held-out split** — load the best checkpoint, re-run
  metric callbacks against the test set without re-training.
- **Generating predictions for downstream analysis** — configure a callback
  with ``save_predictions=True`` (e.g.
  :py:class:`~aero_cfd.callbacks.aero_metrics.AeroMetricsCallback` in
  ``recipes/aero_cfd/``) and write per-sample model outputs to disk for VTK
  export, force-coefficient computation, or reporting.
- **Visualizing model behavior** — re-run plotting/visualization callbacks
  on demand, against any checkpoint.
- **Re-running with different callbacks** — keep the trained model and
  pipeline fixed, swap the callback list to a custom YAML for one-off
  analysis.

Despite the binary name, treat ``noether-eval`` as the generic
inference/evaluation entry point — it has no eval-only logic baked in.

Quick start
-----------

Point ``run_dir`` at the training run output directory (the folder that
contains ``hp_resolved.yaml`` and a ``checkpoints/`` subfolder):

.. code-block:: bash

   noether-eval run_dir=outputs/2026-01-10_abc12

That's the whole minimum invocation. By default it loads the *latest*
checkpoint, reuses every callback that was active during training, and writes
outputs alongside the original training run.

What ``run_dir`` should point at
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A training run is laid out as ``output_path/run_id[/stage_name]``. ``run_dir``
is the deepest directory in that chain — the one that contains
``hp_resolved.yaml``:

.. code-block:: text

   outputs/
     2026-01-10_abc12/        ← run_id
       train/                 ← stage_name (optional)
         hp_resolved.yaml     ← run_dir = outputs/2026-01-10_abc12/train
         checkpoints/
         tracker/

If the run was trained without a ``stage_name``, the ``run_id`` directory
itself is the ``run_dir``.

Overriding the training config
------------------------------

``noether-eval`` is intentionally flexible: the resolved training config is
re-used as the base, and **any** key in it can be overridden on the command
line — no Hydra ``+`` prefix needed, exactly the same syntax as
``noether-train``. A handful of common patterns follow.

Pick a different checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default ``noether-eval`` loads the latest checkpoint. To pick another:

.. code-block:: bash

   # The best checkpoint, by metric. BestCheckpointCallback writes files like
   # `<model>_cp=best_model.<metric_with_slashes_flattened_to_dots>_model.th`,
   # so the tag is `best_model.<metric>`. Look up the configured `metric_key`
   # in the run's hp_resolved.yaml — for `metric_key: loss/test/total`:
   noether-eval run_dir=outputs/2026-01-10_abc12 resume_checkpoint=best_model.loss.test.total

   # An exponential-moving-average snapshot
   noether-eval run_dir=outputs/2026-01-10_abc12 resume_checkpoint=latest_ema=0.9999

   # A specific epoch / update / step
   noether-eval run_dir=outputs/2026-01-10_abc12 resume_checkpoint=E100_U5000_S5000

Send outputs somewhere else
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The run writes outputs to ``output_path/run_id/stage_name`` — by default the
same folder as the training run. Override ``output_path``, ``stage_name`` (or
both) to redirect:

.. code-block:: bash

   # Sibling stage next to the training run
   # → outputs/2026-01-10_abc12/eval/
   noether-eval run_dir=outputs/2026-01-10_abc12/train stage_name=eval

   # Different output root
   # → /scratch/eval_runs/2026-01-10_abc12/train/
   noether-eval run_dir=outputs/2026-01-10_abc12/train output_path=/scratch/eval_runs

   # Both — different root and stage
   noether-eval run_dir=outputs/2026-01-10_abc12/train \
     output_path=/scratch/eval_runs stage_name=eval

In every case ``noether-eval`` still loads the checkpoint from ``run_dir`` —
only the *output* location changes.

Run on different hardware
~~~~~~~~~~~~~~~~~~~~~~~~~

You don't have to re-run on the same accelerator as training. Override
``accelerator`` (and optionally ``devices``):

.. code-block:: bash

   # Apple Silicon
   noether-eval run_dir=outputs/2026-01-10_abc12 accelerator=mps

   # CPU-only spot check
   noether-eval run_dir=outputs/2026-01-10_abc12 accelerator=cpu

   # A specific subset of GPUs
   noether-eval run_dir=outputs/2026-01-10_abc12 devices="0,1"

Switch experiment tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~

Replace the training-time tracker with a disabled or local one for a quick
re-run, or send results to a different W&B project:

.. code-block:: bash

   noether-eval run_dir=outputs/2026-01-10_abc12 tracker=disabled
   noether-eval run_dir=outputs/2026-01-10_abc12 tracker.project=eval-only

Tweak any other training-time key
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same dot-path syntax works for nested keys:

.. code-block:: bash

   # Re-evaluate at a smaller batch size
   noether-eval run_dir=outputs/2026-01-10_abc12 trainer.effective_batch_size=1

   # Point at a different dataset root (e.g. moved data)
   noether-eval run_dir=outputs/2026-01-10_abc12 dataset_root=/new/data/path

Most callbacks honor these overrides because they reuse the same trainer
config keys they did at training time.

Customizing the callbacks
-------------------------

By default ``noether-eval`` reuses the callbacks from training. To add
post-training-only callbacks — whether for extra metrics, prediction saving,
or visualization — write a small YAML and pass it via ``--hp``.

**Evaluation example** — add an offline test-set loss:

.. code-block:: yaml

   # configs/eval_extra.yaml
   trainer:
     callbacks:
       - kind: noether.training.callbacks.OfflineLossCallback
         dataset_key: test
         every_n_epochs: 1

**Inference example** — save denormalized predictions to disk for downstream
analysis (VTK export, force coefficient computation, plotting):

.. code-block:: yaml

   # configs/save_predictions.yaml
   trainer:
     callbacks:
       - kind: aero_cfd.callbacks.AeroMetricsCallback
         dataset_key: test
         every_n_epochs: 1
         forward_properties: ${model.forward_properties}
         save_predictions: true
         predictions_path: ./predictions

Run with either:

.. code-block:: bash

   noether-eval run_dir=outputs/2026-01-10_abc12 --hp configs/eval_extra.yaml
   noether-eval run_dir=outputs/2026-01-10_abc12 --hp configs/save_predictions.yaml

When ``--hp`` is supplied, that file becomes the Hydra base config — use this
escape hatch for power users composing their own inference/eval pipeline. The
``run_dir`` argument and CLI overrides above still work the same way.

**Flipping a flag on an existing callback (no YAML)** — when the change is
just enabling a feature already supported by a configured callback (e.g.
turning on ``save_predictions`` for an ``AeroMetricsCallback`` that was
trained with metrics-only), use a dotted index override. Hydra's ``++``
prefix is required because the keys aren't in the loaded yaml (training left
them at defaults):

.. code-block:: bash

   # Index 4 = position of AeroMetricsCallback in trainer.callbacks
   noether-eval run_dir=outputs/<run_id>/train \
     ++trainer.callbacks.4.save_predictions=true \
     ++trainer.callbacks.4.predictions_path=/path/to/preds

The list index reflects the callback's position in the source's
``hp_resolved.yaml`` ``trainer.callbacks`` list — open the file to confirm.

How it works
------------

``noether-eval`` is a thin post-training callback runner. Under the hood it:

1. Reads ``hp_resolved.yaml`` from ``run_dir``, which captures the full,
   resolved training configuration.
2. Wires that config in as the Hydra base, so every training-time key is a
   valid override target on the command line.
3. Forces ``resume_run_id`` / ``resume_stage_name`` to point at the
   training run, defaults ``resume_checkpoint`` to ``latest``, and applies any
   user overrides on top.
4. Hands the resolved config to :class:`~noether.inference.runners.InferenceRunner`,
   which sets up the trainer/model/tracker the same way training does but uses
   :class:`~noether.core.initializers.PreviousRunInitializer` to load *only*
   the model weights (no optimizer/scheduler state), then calls
   ``trainer.eval(model)`` instead of ``trainer.train(model)``.

``trainer.eval()`` simply iterates the configured callbacks against the
restored weights — there is no separate eval loop, and nothing in the runner
distinguishes "evaluation" from "inference". Whichever callbacks are
configured (metric computation, prediction saving, visualization) decide what
the run actually produces. Callbacks that aren't meaningful here (e.g.
checkpoint saving) are no-ops; callbacks can branch on ``interval_type ==
"eval"`` inside
:py:meth:`~noether.core.callbacks.periodic.PeriodicCallback._periodic_callback`
if they need to behave differently outside of training. See
:py:class:`noether.core.callbacks.periodic.PeriodicCallback` and
:py:class:`noether.core.callbacks.periodic.PeriodicIteratorCallback` for the
callback protocol.

Recipe code on PYTHONPATH
~~~~~~~~~~~~~~~~~~~~~~~~~

If the training run referenced classes from a recipe (e.g.
``aero_cfd.pipeline.AeroMultistagePipeline``), run ``noether-eval`` from a
working directory where those imports resolve — typically the recipe folder
itself, or with ``PYTHONPATH`` set:

.. code-block:: bash

   cd recipes/aero_cfd
   noether-eval run_dir=/path/to/outputs/2026-01-10_abc12 tracker=disabled

Each run directory also contains a ``code.tar.gz`` snapshot of the codebase at
training time, useful when the source tree has drifted.

A note on ``--help`` and the binary name
----------------------------------------

``noether-eval`` is built on Hydra rather than Typer, so ``noether-eval --help``
prints Hydra's generic help text rather than a curated list of arguments —
this guide is the practical reference. The binary is also still spelled
``noether-eval`` for brevity even though the underlying machinery
(:py:class:`~noether.inference.runners.InferenceRunner`, the
``noether.inference`` module) reflects its broader inference + evaluation
scope.
