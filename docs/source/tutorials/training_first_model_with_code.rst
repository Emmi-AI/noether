Training First Model (with Code)
================================

Prerequisites
--------------

- you cloned the **Noether Framework**
- you have a ``tutorial/`` folder in the repo root
- you prepared the ``ShapeNet-Car`` dataset

The fetching and preprocessing instructions are in the ``README.md`` located in the
``src/noether/data/datasets/cfd/shapenet_car/`` folder. Review them first and proceed with the next steps when ready.

What we build in this tutorial
------------------------------

We will build a training run in Python code (no YAML).
The code produces the same config object that the CLI would normally create.

You will learn:

- how the config is structured (datasets, model, trainer, callbacks)
- what dataset stats/specs and normalizers do
- how to start training with ``HydraRunner().main()``

Implementation
--------------

Overview
~~~~~~~~

Sometimes you want to run training via code. We get it.

Previously we covered :doc:`how to train using CLI and configs <training_first_model_with_configs>`, now we will focus
on making things to work via the Python code.

Relevant files for this can be found under ``src/noether/training/`` folder. Let's briefly go over it:

- ``training/callbacks`` - callbacks executed during and post training
- ``training/cli/`` - the CLI definition that we use in the :doc:`previous tutorial <training_first_model_with_configs>`
- ``training/runners/`` - core ``Hydra`` runner logic that we will use to make the code to work
- ``training/trainers/`` - ``BaseTrainer`` that can be extended downstream, used directly in the ``Hydra`` runner

In this tutorial we skip ``yaml`` configs and build the run config in Python.

.. note::

    The example code below uses typed configs and many small schema classes. If this looks overwhelming, don't worry:
    it will make sense as you follow the steps.

Step 1: Create an entry point
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's create a new file ``tutorial/train_shapenet_upt.py`` and use it to run our pipeline. Why "tutorial"? Because
it has necessary components to get us started and here we want to see a difference between "configs vs. code" workflows.

Step 2: Create necessary imports
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most schema classes live in ``noether.core.schemas``. They help us keep typing consistent and validate inputs at
runtime. These configs are Pydantic models, so if something is wrong you will get a clear validation error.

.. code-block:: python

    from __future__ import annotations

    from pathlib import Path
    from typing import Any, Literal, Sequence

    import torch

    from noether.core.configs import StaticConfig
    from noether.core.schemas.callbacks import (
        BestCheckpointCallbackConfig,
        CheckpointCallbackConfig,
        EmaCallbackConfig,
        OfflineLossCallbackConfig,
    )
    from noether.core.schemas.dataset import AeroDataSpecs, CAEMLDatasetConfig, DatasetBaseConfig, DatasetWrappers, RepeatWrapperConfig
    from noether.core.schemas.modules import (
        DeepPerceiverDecoderConfig,
        SupernodePoolingConfig,
        PerceiverBlockConfig,
        TransformerBlockConfig,
    )
    from noether.core.schemas.normalizers import AnyNormalizer, MeanStdNormalizerConfig, PositionNormalizerConfig
    from noether.core.schemas.optimizers import OptimizerConfig
    from noether.core.schemas.schedules import LinearWarmupCosineDecayScheduleConfig
    from noether.core.schemas.schema import ConfigSchema, StaticConfigSchema
    from noether.core.schemas.statistics import AeroStatsSchema
    from noether.training.runners import HydraRunner
    from tutorial.callbacks.surface_volume_evaluation_metrics import (
        SurfaceVolumeEvaluationMetricsCallbackConfig,
    )
    from tutorial.schemas.models.upt_config import UPTConfig
    from tutorial.schemas.pipelines.aero_pipeline_config import AeroCFDPipelineConfig
    from tutorial.schemas.trainers.automotive_aerodynamics_trainer_config import AutomotiveAerodynamicsCfdTrainerConfig

Let's go over each group to better understand the outline:

Data:

- ``noether.core.configs`` - the main config classes, contains configuration for all other components
- ``noether.core.schemas.dataset`` - dataset related configs and types for type-hinting
- ``noether.core.schemas.modules`` - building blocks of our models, we will use UPT architecture
- ``noether.core.schemas.normalizers`` - data normalization configs to be applied during the data loading
- ``noether.core.schemas.statistics`` - data statistics aggregation, e.g. mean, std, etc.
- etc.

Training:

- ``noether.core.schemas.callbacks`` - relevant callbacks for our training
- ``noether.core.schemas.optimizers`` - optimizer config
- ``tutorial.schemas.trainers.automotive_aerodynamics_trainer_config`` - trainer configuration
- ``tutorial.schemas.models`` - configs for model initialization
- etc.

Execution:

- ``noether.training.runners`` - orchestrators responsible for pipeline execution

.. note::
    ``HydraRunner`` from ``noether`` comes with a few public methods: ``run()`` is used by the CLI, and ``main()``
    can be used to run the pipeline via Python. We use the latter to avoid YAML files in this tutorial.

Step 3: Declare main() function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def main() -> None:
        dataset_root = Path("/Users/user/datasets/shapenet_car")  # feel free to change this to match your structure
        output_path = dataset_root / "outputs"

        data_specs = build_specs()
        dataset_normalizer = build_dataset_normalizer()

        model_forward_properties = [
            "surface_mask_query",
            "surface_position_batch_idx",
            "surface_position_supernode_idx",
            "surface_position",
            "surface_query_position",
            "volume_query_position",
        ]

        upt_model_config = build_model_config(data_specs, model_forward_properties)
        aero_trainer_config = build_trainer_config(data_specs, model_forward_properties)

        HydraRunner().main(
            device=torch.device("mps"),
            config=ConfigSchema(...),  # note that '...' is a placeholder that we will populate later
            output_path=output_path.as_posix(),
        )

    if __name__ == "__main__":
        main()

This is the core outline of the pipeline. If you will run this code you will get a lot of errors, but conceptually we
are ready.

Worth noting that we have several functions that start with ``build_*`` - we will create them in a minute, but first
let's take a look at what we have:

1. We declared input and output directories for our training.
2. We will use python-based configs to define our pipeline. These configs will be created via ``build_*`` methods or
   directly in the ``HydraRunner().main(config=ConfigSchema(...))`` declaration.
3. We declare model properties that will be sent to the forward pass.
4. The final execution will be handled by ``noether``'s internal runner.


Step 4: Dataset configs
~~~~~~~~~~~~~~~~~~~~~~~

Now we will declare dataset constants and convenience ``build_`` methods (you can place it right under the imports):

.. code-block:: python

    DATASET_STATS = {
        "raw_pos_min":  [-4.5],
        "raw_pos_max": [6.0],
        "surface_pressure_mean": [-36.4098],
        "surface_pressure_std": [48.6757],
        "volume_velocity_mean": [0.00293915, -0.0230546, 17.546032],
        "volume_velocity_std": [1.361689, 1.267649, 5.850353],
        "volume_sdf_mean": [3.74222e-01],
        "volume_sdf_std": [1.78948e-01],
    }
    DATA_SPECS = {
        "position_dim": 3,
        "surface_feature_dim": {
            "surface_sdf": 1,
            "surface_normals": 3,
        },
        "volume_feature_dim": {
            "volume_sdf": 1,
            "volume_normals": 3,
        },
        "surface_output_dims": {
            "pressure": 1,
        },
        "volume_output_dims":{
            "velocity": 3,
        },
    }


    def build_stats() -> AeroStatsSchema:
        return AeroStatsSchema(**DATASET_STATS)


    def build_specs() -> AeroDataSpecs:
        return AeroDataSpecs(**DATA_SPECS)


.. code-block:: python

    def build_dataset_config(
        mode: Literal["train", "test"],
        dataset_root: str,
        data_specs: dict[str, Any] | AeroDataSpecs,
        dataset_statistics: dict[str, Sequence[float]],
        dataset_normalizer: dict[str, list[AnyNormalizer]],
        dataset_wrappers: list[DatasetWrappers] | None = None,
    ) -> CAEMLDatasetConfig:
        return CAEMLDatasetConfig(
            kind="noether.data.datasets.cfd.ShapeNetCarDataset",
            root=dataset_root,
            pipeline=AeroCFDPipelineConfig(
                kind="tutorial.pipeline.AeroMultistagePipeline",
                num_surface_points=3586, # max = 3586
                num_volume_points=4096,   # max = 28504
                num_surface_queries=3586,
                num_volume_queries=4096,
                num_supernodes=3586,
                sample_query_points=False,
                use_physics_features=False,
                dataset_statistics=AeroStatsSchema(**dataset_statistics),
                data_specs=data_specs if isinstance(data_specs, AeroDataSpecs) else AeroDataSpecs(**data_specs),
            ),
            split=mode,
            dataset_normalizers=dataset_normalizer,
            dataset_wrappers=dataset_wrappers,
            included_properties=None,
            excluded_properties={"surface_friction", "volume_pressure", "volume_vorticity"},
        )

This config defines our datasets (both ``train`` and ``test``). Note the ``kind`` fields: they are strings that
point to a Python class path inside the codebase. The factory uses them to build real objects, just like in the
config-driven workflow.

Step 5: Trainer config
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def build_trainer_config(model_forward_properties: list[str]) -> AutomotiveAerodynamicsCfdTrainerConfig:
        batch_size = 1
        loss_and_log_every_n_epochs = 1
        save_and_ema_every_n_epochs = 10

        return AutomotiveAerodynamicsCfdTrainerConfig(
            kind="tutorial.trainers.AutomotiveAerodynamicsCFDTrainer",
            surface_weight=1.0,
            volume_weight=1.0,
            surface_pressure_weight=1.0,
            volume_velocity_weight=1.0,
            use_physics_features=False,
            precision="float32",
            max_epochs=500,
            effective_batch_size=batch_size,
            log_every_n_epochs=loss_and_log_every_n_epochs,
            callbacks=[
                CheckpointCallbackConfig(
                    kind="noether.core.callbacks.CheckpointCallback",
                    save_weights=True,
                    save_latest_weights=True,
                    save_latest_optim=False,
                    every_n_epochs=save_and_ema_every_n_epochs,
                ),
                # validation loss
                OfflineLossCallbackConfig(
                    kind="noether.training.callbacks.OfflineLossCallback",
                    batch_size=batch_size,
                    every_n_epochs=loss_and_log_every_n_epochs,
                    dataset_key="test",
                ),
                BestCheckpointCallbackConfig(
                    kind="noether.core.callbacks.BestCheckpointCallback",
                    every_n_epochs=batch_size,
                    metric_key="loss/test/total",
                ),
                # test loss
                SurfaceVolumeEvaluationMetricsCallbackConfig(
                    kind="tutorial.callbacks.SurfaceVolumeEvaluationMetricsCallback",
                    batch_size=1,
                    every_n_epochs=loss_and_log_every_n_epochs,
                    dataset_key="test",
                    forward_properties=model_forward_properties,
                ),
                SurfaceVolumeEvaluationMetricsCallbackConfig(
                    kind="tutorial.callbacks.SurfaceVolumeEvaluationMetricsCallback",
                    batch_size=1,
                    every_n_epochs=500,
                    dataset_key="test_repeat",
                    forward_properties=model_forward_properties,
                ),
                # ema
                EmaCallbackConfig(
                    kind="noether.core.callbacks.EmaCallback",
                    every_n_epochs=save_and_ema_every_n_epochs,
                    save_weights=False,
                    save_last_weights=False,
                    save_latest_weights=True,
                    target_factors={0.9999},
                ),
            ],
            forward_properties=model_forward_properties,
            target_properties=[
                "surface_pressure_target",
                "volume_velocity_target",
            ],
        )

Trainer configuration is at the core of the training pipeline, as you can see the callbacks is the crucial component
of it.

Step 6: Filling the gaps
~~~~~~~~~~~~~~~~~~~~~~~~

At last, we will populate the placeholder fields that we declared for the ``HydraRunner``:

.. code-block:: python

    HydraRunner().main(
        device=torch.device("mps"),
        config=ConfigSchema(
            name=None,
            accelerator="mps",  # can be "cpu", "gpu", "mps"
            stage_name="train",
            dataset_kind="noether.data.datasets.cfd.ShapeNetCarDataset",
            dataset_root=dataset_root.as_posix(),
            resume_run_id=None,
            resume_stage_name=None,
            resume_checkpoint=None,
            seed=42,
            dataset_statistics=DATASET_STATS,
            dataset_normalizer=dataset_normalizer,
            output_path=output_path.as_posix(),
            tracker=None,
            run_id=None,
            devices=None,
            num_workers=None,
            datasets={
                "train": build_dataset_config(
                    mode="train",
                    dataset_root=dataset_root.as_posix(),
                    data_specs=data_specs,
                    dataset_statistics=DATASET_STATS,
                    dataset_normalizer=dataset_normalizer,
                    dataset_wrappers=None,
                ),
                "test": build_dataset_config(
                    mode="test",
                    dataset_root=dataset_root.as_posix(),
                    data_specs=data_specs,
                    dataset_statistics=DATASET_STATS,
                    dataset_normalizer=dataset_normalizer,
                    dataset_wrappers=None,
                ),
                "test_repeat": build_dataset_config(
                    mode="test",
                    dataset_root=dataset_root.as_posix(),
                    data_specs=data_specs,
                    dataset_statistics=DATASET_STATS,
                    dataset_normalizer=dataset_normalizer,
                    dataset_wrappers=[RepeatWrapperConfig(
                        kind="noether.data.base.wrappers.RepeatWrapper",
                        repetitions=10,
                    )],
                ),
            },
            model=upt_model_config,
            trainer=aero_trainer_config,
            debug=False,
            store_code_in_output=False,
            output_path=output_path.as_posix(),
        ),
    )

As you can see above, there are multiple arguments that were defined with ``None``. They are present here to show
available settings that you can modify to your needs. You can also freely remove them from the code to make a bit more
lightweight. This won't break the logic.

Step 7: Run training
~~~~~~~~~~~~~~~~~~~~

You are ready to start the training! If you are using an IDE - simply run the file. Otherwise, in the repo root
from your terminal:

.. code-block:: bash

    uv run python -m tutorial.train_shapenet_upt

This makes Python add the repo root to sys.path, so ``from tutorial.*`` works. Alternatively, you can add repo root
to PYTHONPATH:

.. code-block:: bash

    PYTHONPATH=. uv run python tutorial/train_shapenet_upt.py

If everything is set up correctly, you should see the logs indicating successful initialization and training
(use your task manager and/or activity monitor to see if the hardware is properly utilized).

The output directory will be populated with files like this:

.. code-block:: bash

    shapenet_car/outputs/YYYY-MM-DD_<SHORT_ID>
    └── train
        ├── basetracker
        │   └── config.yaml
        ├── hp_resolved.yaml
        └── log.txt

After the training progresses, check this folder again - you will find the checkpoints and other training artifacts
there.
