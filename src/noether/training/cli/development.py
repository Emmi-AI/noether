#  Copyright © 2026 Emmi AI GmbH. All rights reserved.

import os
import random
import sys

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from noether.core.callbacks import PeriodicCallback
from noether.core.factory import DatasetFactory, Factory, class_constructor_from_class_path
from noether.core.schemas.schema import ConfigSchema
from noether.data.container import DataContainer
from noether.training.cli import setup_hydra

_HELP_TEXT = """\
Noether Development Runner
--------------------------
Step-by-step build, test and debug your dataset, pipeline, model forward pass, and
training step, and callbacks against a single sampled batch — without launching a full
training run.

What it does, in order:
  1. Dataset      — instantiates the dataset and samples batch_size items.
  2. Pipeline     — collates the raw samples through the configured pipeline
                    (if defined) and shows shapes/dtypes before and after.
  3. Model        — runs a forward pass and prints the output structure.
  4. Trainer      — runs a single train_step, executes backward(), and
                    reports the total loss, per-loss breakdown, and which
                    parameters received gradients.
  5. Callback     — if exactly one PeriodicCallback is configured, calls
                    process_data() and prints its output.

  Each stage is skipped gracefully when not present in the config.

Usage:
  noether-development <config.yaml> [overrides]
  noether-development --hp <config.yaml> [overrides]

Arguments:
  config.yaml   Path to a development YAML config file.
  --hp          Alternative flag for specifying the config path.
  overrides     Hydra-style key=value overrides, e.g. batch_size=4.

Examples:
  noether-development --hp development/configs/development_config.yaml
"""
if "--help" in sys.argv or "-h" in sys.argv:
    print(_HELP_TEXT)
    sys.exit(0)
setup_hydra()


def _describe_value(value):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return {
                "type": "Tensor",
                "shape": tuple(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
    except ImportError:
        pass
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return {"type": "ndarray", "shape": value.shape, "dtype": str(value.dtype)}
    except ImportError:
        pass
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "len": len(value)}
    return {"type": type(value).__name__}


def log_batch(batch, title="BATCH CONTENTS"):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"Batch type: {type(batch)}")
    if isinstance(batch, dict):
        print(f"Batch keys ({len(batch)}): {list(batch.keys())}")
        for key, value in batch.items():
            print(f"\n  [{key}]")
            print(f"    type: {type(value).__name__}")
            try:
                import torch

                if isinstance(value, torch.Tensor):
                    print(f"    shape:  {tuple(value.shape)}")
                    print(f"    dtype:  {value.dtype}")
                    print(f"    device: {value.device}")

                    if value.numel() <= 20:
                        print(f"    values: {value}")
                    else:
                        print(f"    first 10 values: {value.flatten()[:10].tolist()}")
                    continue
            except ImportError:
                pass
            try:
                import numpy as np

                if isinstance(value, np.ndarray):
                    print(f"    shape:  {value.shape}")
                    print(f"    dtype:  {value.dtype}")
                    if np.issubdtype(value.dtype, np.floating):
                        print(f"    min/max/mean: {value.min():.4f} / {value.max():.4f} / {value.mean():.4f}")
                    else:
                        print(f"    min/max: {value.min()} / {value.max()}")
                    if value.size <= 20:
                        print(f"    values: {value}")
                    else:
                        print(f"    first 10 values: {value.flatten()[:10].tolist()}")
                    continue
            except ImportError:
                pass
            if isinstance(value, (list, tuple)):
                print(f"    len:    {len(value)}")
                print(f"    first 5 items: {value[:5]}")
            elif isinstance(value, str):
                print(f"    value:  {value!r}")
            else:
                print(f"    value:  {value}")
    else:
        print(f"Batch value: {batch}")
    print("=" * 60 + "\n")


def log_batch_diff(batch, collated_batch):
    print("\n" + "=" * 60)
    print("BATCH vs COLLATED BATCH DIFF")
    print("=" * 60)
    if not isinstance(batch, dict) or not isinstance(collated_batch, dict):
        print("  (cannot diff non-dict batches)")
        print("=" * 60 + "\n")
        return

    raw_keys = set(batch.keys())
    col_keys = set(collated_batch.keys())

    only_in_raw = raw_keys - col_keys
    only_in_col = col_keys - raw_keys
    common = raw_keys & col_keys

    if only_in_raw:
        print(f"\n  Keys only in raw batch: {sorted(only_in_raw)}")
    if only_in_col:
        print(f"\n  Keys only in collated batch: {sorted(only_in_col)}")

    print(f"\n  Common keys ({len(common)}):")
    for key in sorted(common):
        raw_desc = _describe_value(batch[key])
        col_desc = _describe_value(collated_batch[key])
        changed = raw_desc != col_desc
        marker = "  *" if changed else "   "
        print(f"{marker} [{key}]")
        if changed:
            print(f"      raw:      {raw_desc}")
            print(f"      collated: {col_desc}")
        else:
            print(f"      {col_desc}  (unchanged)")

    print("=" * 60 + "\n")


@hydra.main(
    config_path=None,
    config_name=None,
    version_base="1.3",
)
def main(hydra_config: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    # add working directory to PYTHONPATH
    sys.path.insert(0, hydra.utils.get_original_cwd())  # store this

    raw_config = yaml.safe_load(OmegaConf.to_yaml(hydra_config, resolve=True))
    # get config schema
    config_schema = class_constructor_from_class_path(hydra_config["config_schema_kind"])

    config: ConfigSchema = config_schema(**raw_config)

    assert len(config.datasets) == 1, "Only able to handle one dataset for development"
    dataset_key = next(iter(config.datasets.keys()))
    dataset_config = config.datasets[dataset_key]
    dataset = DatasetFactory().create(dataset_config)

    assert dataset is not None

    perm = random.sample(range(len(dataset)), config.development_batch_size)  # type: ignore[attr-defined]
    batch = []
    for i in perm:
        sample = dataset[i]
        batch.append(sample)

    log_batch(batch[0], title="RAW BATCH SAMPLE CONTENTS (first random sample from the batch)")

    if dataset_config.pipeline is not None:
        pipeline = Factory().create(dataset_config.pipeline)
        assert pipeline is not None
        collated_batch = pipeline(batch)

        log_batch(collated_batch, title="COLLATED BATCH CONTENTS")

    else:
        print(
            "\nNo pipeline defined for dataset, skipping collation step. For all additional steps (model forward pass, trainer step, callback step), a pipeline is required."
        )
        return

    if config.model is not None:
        datasets = {dataset_key: dataset}

        model = Factory().instantiate(
            config.model,
        )

        forward_batch = {key: collated_batch[key] for key in config.model.forward_properties if key in collated_batch}
        out = model(**forward_batch)
        log_batch(out, title="MODEL OUTPUT CONTENTS")

        if config.trainer is not None:
            data_container = DataContainer(datasets=datasets, num_workers=config.num_workers, pin_memory=False)

            trainer = Factory().create(
                config.trainer,
                data_container=data_container,
                device="cpu",
                tracker=None,
                path_provider=None,
                metric_property_provider=None,
            )
            assert trainer is not None
            trainer_output = trainer.train_step(model=model, batch=collated_batch)

            try:
                trainer_output.total_loss.backward()  # just to check that the loss is properly connected to the model outputs and that backward works without error
                print("\nBackward pass completed successfully.")
            except Exception as e:
                print(f"\nError during backward pass: {e}")

            if all(p.grad is not None for p in model.parameters()):
                print("\nAll model parameters received gradients successfully.")
            else:
                no_grad_params = [name for name, param in model.named_parameters() if param.grad is None]
                print(f"\nWarning: The following parameters did not receive gradients: {no_grad_params}")

            print("\n" + "=" * 60)
            print("TRAINER OUTPUT")
            print("=" * 60)
            print(f"  total_loss: {trainer_output.total_loss.item():.6f}")
            if trainer_output.losses_to_log:
                print(f"\n  losses_to_log ({len(trainer_output.losses_to_log)}):")
                for k, v in trainer_output.losses_to_log.items():
                    print(f"    [{k}]: {v.item():.6f}")
            if trainer_output.additional_outputs:
                print("\n  additional_outputs:")
                log_batch(trainer_output.additional_outputs, title="TRAINER ADDITIONAL OUTPUTS")
            else:
                print("\n  additional_outputs: None")
            print("=" * 60 + "\n")

            if config.trainer.callbacks and len(config.trainer.callbacks) == 1:
                callback = Factory().instantiate(
                    config.trainer.callbacks[0],
                    model=model,
                    trainer=trainer,
                    data_container=data_container,
                    tracker=None,
                    log_writer=None,
                    checkpoint_writer=None,
                    metric_property_provider=None,
                    development=True,
                )

                if isinstance(callback, PeriodicCallback):
                    perm = random.sample(
                        range(len(data_container.datasets[dataset_key])), config.trainer.callbacks[0].batch_size
                    )  # type: ignore[attr-defined]
                    batch = []
                    for i in perm:
                        sample = data_container.datasets[dataset_key][i]
                        batch.append(sample)
                    collated_batch = pipeline(batch)
                    out_callback = callback.process_data(collated_batch)  # type: ignore[attr-defined]
                    log_batch(out_callback, title="CALLBACK OUTPUT CONTENTS")
                else:
                    print(
                        f"\nCallback is not a PeriodicCallback (but {type(callback).__name__}), so skipping callback step since the development callback is currently only implemented for PeriodicCallbacks."
                    )
            else:
                if not config.trainer.callbacks or len(config.trainer.callbacks) == 0:
                    print("\nNo callbacks defined in trainer, skipping callback step.")
                else:
                    print(
                        "\nMultiple callbacks defined in trainer, skipping callback step for now since the development callback can only done by one."
                    )
        else:
            trainer = None

    else:
        print(
            "\nNo model defined in config, skipping model instantiation and forward pass. Also skipping trainer and callback steps since they depend on the model."
        )


if __name__ == "__main__":
    main()
