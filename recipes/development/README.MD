# Development Recipe

This recipe provides a minimal example of how to build module in Noether use with the `noether-development` CLI.
It is intended as a starting point when developing new components in isolation, without needing a full training setup.

## What's included

| File | Module |
|---|---|
| `dataset/development_dataset.py` | `DevelopmentDataset` — Dataset that generates random dummy data |
| `pipeline/development_pipeline.py` | `DevelopmentPipeline` — Multi-stage pipeline that collates the datapoints into a batch |
| `model/development_model.py` | `DevelopmentModel` — Two-layer MLP model |
| `trainer/development_trainer.py` | `DevelopmentTrainer` — Trainer that computes MSE loss |
| `callbacks/development_callback.py` | `DevelopmentCallback` — Periodic MSE callback |
| `configs/development_config.yaml` | Config file wiring everything together |

## Running the CLI

```bash
uv run noether-development --hp recipes/development/configs/development_config.yaml
```

The CLI prints the raw batch content, collated batch, model output, trainer loss, and callback output in sequence.
Individual modules (dataset, pipeline, model, trainer, callbacks) can be commented out in the config to test components in isolation.

## Going further

For a step-by-step walkthrough of how each module is built and configured, see the full guide:
[`docs/source/guides/training/development_cli.rst`](../../docs/source/guides/training/development_cli.rst)

Once all components are developed, a full training run can be launched with (if some additional configs are added, see the docs for all the details):

```bash
uv run noether-train --hp recipes/development/configs/development_config.yaml +seed=1 +devices="0" +accelerator=cpu
```
