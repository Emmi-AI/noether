<div align="center">

# The Emmi-AI/noether Tutorial 

**Authors**: Maurits Bleeker, Markus Hennerbichler, and Pavel Kuksa

For questions, please create an issue in our [GitHub repo](https://github.com/Emmi-AI/noether/issues).
</div>

# Table of Contents

- [Introduction](#introduction)
  - [File structure of the project](#file-structure-of-the-project)
    - [Core components](#core-components)
  - [Project setup](#project-setup)
    - [Boilerplate project](#boilerplate-project)
- [Tutorial](#tutorial)
  - [Configuration (`configs/`)](#configuration-configs)
    - [Configuration architecture](#configuration-architecture)
    - [Example: ShapeNet-Car configuration](#example-shapenet-car-configuration)
    - [Defining experiment configurations](#defining-experiment-configurations)
    - [Running experiments](#running-experiments)
    - [Pydantic schemas for type safety](#pydantic-schemas-for-type-safety)
    - [Example: Transformer configuration schema](#example-transformer-configuration-schema)
    - [From schema to YAML](#from-schema-to-yaml)
    - [Tutorial configuration schema](#tutorial-configuration-schema)
  - [Object instantiation](#object-instantiation)
  - [The Dataset (`datasets/`)](#the-dataset-datasets)
    - [The `getitem_*` pattern: Modular data loading](#the-getitem_-pattern-modular-data-loading)
    - [ShapeNet-Car dataset structure](#shapenet-car-dataset-structure)
    - [Dataset configuration](#dataset-configuration)
    - [Selective data loading with `excluded_properties`](#selective-data-loading-with-excluded_properties)
    - [Essential dataset methods](#essential-dataset-methods)
    - [Tensor normalization with decorators](#tensor-normalization-with-decorators)
    - [Computing dataset statistics](#computing-dataset-statistics)
    - [Noether dataset zoo](#noether-dataset-zoo)
  - [The Multi-Stage Pipeline (`pipeline/`)](#the-multi-stage-pipeline-pipeline)
    - [Sample processors](#sample-processors)
    - [Collators](#collators)
    - [Batch processors](#batch-processors)
  - [The Trainer (`trainers/`)](#the-trainer-trainers)
    - [BaseTrainer implementation](#basetrainer-implementation)
    - [BaseTrainer configuration](#basetrainer-configuration)
    - [AutomotiveAerodynamicsCFDTrainer implementation](#automotiveaerodynamicscfdtrainer-implementation)
  - [Models (`models/`)](#models-models)
    - [The `ModelBaseConfig` schema](#the-modelbaseconfig-schema)
    - [Implementing a custom model](#implementing-a-custom-model)
    - [The `BaseModel` for standardization](#the-basemodel-for-standardization)
    - [Handling model outputs](#handling-model-outputs)
    - [Composite Models](#composite-models)
    - [`Noether` model zoo](#noether-model-zoo)
  - [Callbacks (`callbacks/`)](#callbacks-callbacks)
    - [Overview](#overview)
    - [Implementing PeriodicIteratorCallback](#implementing-periodiciteratorcallback)
    - [Configuring callbacks](#configuring-callbacks)
    - [Schema validation for callbacks](#schema-validation-for-callbacks)
    - [Denormalization for metrics](#denormalization-for-metrics)
    - [Computed metrics](#computed-metrics)
    - [Final evaluation with repeated testing](#final-evaluation-with-repeated-testing)
  - [Running the experiments](#running-the-experiments)
    - [Running SLURM jobs](#running-slurm-jobs)
    - [Running a single experiment](#running-a-single-experiment)
    - [Running multi-GPU experiments](#running-multi-gpu-experiments)
    - [Running Inference](#running-inference)
  - [Extra utilities and tips](#extra-utilities-and-tips)


# Introduction
Welcome to the `Noether Framework` tutorial!

This tutorial demonstrates how to use the `Noether Framework` through a practical project based on the experiments from Section 4.4 of the [AB-UPT paper](https://arxiv.org/pdf/2502.09692). 
While this tutorial covers the core functionality of the framework, it does not cover every possible feature or use case.

> [!NOTE]
> This tutorial presents recommended practices and patterns, but it is not a full blueprint how to use the framework. 
The `Noether Framework` is flexible and supports multiple approaches to implementing the same functionality.

## File structure of the project

The tutorial project follows the following directory structure:

```
└── callbacks/        # Callbacks for evaluation, logging, and monitoring during training
└── configs/          # YAML files for configuring experiments using Hydra
    └── callbacks/
    └── data_specs/
    └── dataset_normalizers/
    └── dataset_statistics/
    └── datasets/
    └── experiments/
    └── model/
    └── optimizer/
    └── pipeline/
    └── tracker/
    └── trainer/
    train_ahmedml.yaml
    train_caeml.yaml
    train_drivaerml.yaml
    train_drivaernet.yaml # Additional dataset (not covered in tutorial)
    train_shapenet.yaml  
    train_wing.yaml   # Additional dataset (not covered in tutorial)
└── jobs/             # SLURM job scripts for running experiments on clusters
└── model/            # Model architecture definitions
└── pipeline/         # Data processing and collation pipeline
└── schemas/          # Pydantic schemas for configuration validation
    └── callbacks/
    └── datasets/
    └── models/
    └── pipelines/
    └── trainers/
    config_schema.py
└── trainers/         # Trainer classes that manage the training loop
```

**Minimal required structure** for any `Noether` project:

```
└── callbacks/        # Can be empty if only using default callbacks
└── configs/          # Required: defines all configurations
└── datasets/         # Required only for custom datasets
└── pipeline/         # Required: defines data processing
└── model/            # Required: defines model architectures
└── trainers/         # Required: defines training logic
```

The [`configs/`](./configs) directory roughly mirrors the root folder structure; for each module or class defined in the project, there is a corresponding configuration file. 
This organizational pattern makes it easy to locate and manage configurations.

### Core components

Every `Noether` project consists of the following core modules (in alphabetical order):

1. **Callbacks**: Classes that compute metrics and statistics at specific points during training. Can be empty when using only the framework's default callbacks.
2. **Configs**: YAML configuration files that define all hyperparameters, paths, and settings for the training pipeline.
3. **Dataset**: Provides the interface between raw data on disk and the multi-stage pipeline. Defines how to load individual tensors for each data sample. This tutorial uses pre-implemented datasets, but you can create custom ones.
4. **Model**: Defines the model architecture and its forward pass.
5. **Pipeline**: Defines the multi-stage data pipeline that loads, processes, and collates individual samples into batches for training.
6. **Schemas**: Pydantic schemas that define the input data of each class we define in our project.
7. **Trainer**: The trainer loop takes batches from the pipeline, runs the model's forward pass and computes the loss. 

## Project setup

**Prerequisites**: Python 3.12

Clone the repository and set up the environment:

```bash
git clone https://github.com/Emmi-AI/noether.git
cd noether/tutorial
uv venv --python 3.12
source .venv/bin/activate
uv pip install emmiai-noether
```

If the built package is not available, you can build it from source:

```bash
uv pip install ..
```

### Boilerplate project

The [`boilerplate_project/`](../boilerplate_project) directory contains a minimal working example that demonstrates the essential components needed for training in the `Noether Framework`. This stripped-down project is useful for:
- Understanding the minimum required code for each module
- Quick prototyping of new projects
- Reference when building your own `Noether` applications

We recommend reviewing the boilerplate project alongside this tutorial to see what minimal implementations look like.

To run the boilerplate project, use the following command: 
```bash
uv run noether-train --hp ./boilerplate_project/configs/base_experiment.yaml +seed=1 +devices=\"0\"
```

> [!IMPORTANT]
> The `Noether Framework` by defaults runs on GPU. If no GPU is available, please add either `+accelerator=cpu` or `+accelerator=mps` to the command.

# Tutorial

This tutorial covers each module of the `Noether Framework` in a logical progression. 
While we have organized the content to flow naturally, you may need to reference earlier or later sections every now and then, as some components interact with multiple parts of the pipeline.

## Configuration ([`configs/`](./configs/))

> [!NOTE]
> This section assumes familiarity with [Hydra](https://hydra.cc/docs/intro/) configuration management and [Pydantic](https://docs.pydantic.dev/latest/) schemas. If you're new to these tools, we recommend reviewing their official documentation before proceeding.

The configuration is the backbone of the `Noether Framework`, enabling reproducible, modular, and type-safe experiment definitions. 
All experiments are defined through **YAML configuration files** that use:

- **Hydra** for hierarchical composition and command-line overrides
- **Pydantic** for runtime data validation and type safety

### Configuration architecture

The `Noether Framework` uses a **hierarchical configuration pattern** where:

1. **Base configurations** define default settings for each component (datasets, models, trainers, etc.)
2. **Experiment configurations** compose and override base configs for specific experiments
3. **Command-line overrides** allow quick parameter sweeps without file changes

The main entry point for any experiment is a top-level configuration file like [**`train_shapenet.yaml`**](./configs/train_shapenet.yaml), which serves as the composition root that brings together all required components.

### Example: ShapeNet-Car configuration

[**`train_shapenet.yaml`**](./configs/train_shapenet.yaml) demonstrates the structure of a complete experiment configuration. 
Let's break down its key components:

```yaml
# @package _global_

# Define key values here that are used multiple times in the config files.
dataset_root: <path to your shapenet dataset root>
dataset_kind: noether.data.datasets.cfd.ShapeNetCarDataset # class path to the dataset
config_schema_kind: tutorial.schemas.config_schema.TutorialConfigSchema # class path to the config schema in your down stream project
excluded_properties: 
  - surface_friction
  - volume_pressure
  - volume_vorticity

defaults:
  - data_specs: shapenet_car
  - dataset_normalizers: shapenet_dataset_normalizers
  - dataset_statistics: shapenet_car_stats
  - model: ??? # Intentionally undefined - specified per experiment
  - trainer: shapenet_trainer
  - datasets: shapenet_dataset
  - tracker: ??? # Intentionally undefined - specified per experiment
  - callbacks: training_callbacks_shapenet
  - pipeline: shapenet_pipeline
  - optimizer: adamw
  - _self_
```

Each entry like `dataset_normalizers: shapenet_dataset_normalizers` tells Hydra to load `configs/dataset_normalizers/shapenet_dataset_normalizers.yaml` and merge it into the final configuration.

The `???` marker indicates required fields that must be specified in experiment configs. 
The `_self_` marker controls when the current file's values override inherited ones (placing it last gives current file highest priority).

**Complete configuration structure:**

To run an experiment, you need configurations for:

1. **Model**: Architecture and hyperparameters
2. **Trainer**: Trainer config 
3. **Callbacks**: Evaluation, logging, and monitoring
4. **Tracker**: Experiment tracking (W&B or disabled)
5. **Dataset(s)**: Dataset config
6. **Pipeline**: Data preprocessing and collation
7. **Optimizer**: Optimization algorithm

Most components remain constant across experiments on the same dataset. 
For example, when training different models on ShapeNet-Car, only the `model` and `tracker` configurations typically change, while `dataset`, `pipeline`, `trainer`, and `callbacks` remain fixed.

**Example: Dataset configuration**

The base dataset configuration [`configs/datasets/shapenet_dataset.yaml`](./configs/datasets/shapenet_dataset.yaml) demonstrates config composition:

```yaml
train:
  root: ${dataset_root}
  kind: ${dataset_kind}
  split: train
  pipeline: ${pipeline}
  dataset_normalizers: ${dataset_normalizers}
  excluded_properties: ${excluded_properties}
test:
  root: ${dataset_root}
  kind: ${dataset_kind}
  split: test
  pipeline: ${pipeline}
  dataset_normalizers: ${dataset_normalizers}
  excluded_properties: ${excluded_properties}
```

Notice the `${variable_name}` references? These resolve to values defined in the top-level `train_shapenet.yaml`. This pattern avoids duplication: `dataset_root` is defined once, used everywhere. 
To make the training work, add the `dataset_root` folder where the preprocessed data is stored to `train_shapenet.yaml`.
To preprocess the data, have a look at [preprocessing.py)](../src/noether/data/datasets/cfd/shapenet_car/preprocessing.py) of the ShapeNet-Car dataset.

**Config groups and directory structure:**

The `configs/` directory roughly mirrors the component structure:

```
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
└── experiments/                 # Experiment-specific overrides
    └── shapenet/
        ├── transformer.yaml
        ├── upt.yaml
        └── ...
```

When you specify `datasets: shapenet_dataset` in the defaults list, Hydra automatically loads `configs/datasets/shapenet_dataset.yaml`.

### Defining experiment configurations

Experiment-specific configurations compose base configs and apply targeted overrides. An experiment file should:

1. Select a specific model variant
2. Choose a tracker (W&B or disabled)
3. Override any experiment-specific hyperparameters

**Example: Transformer experiment**

The Transformer experiment configuration [**`configs/experiments/shapenet/transformer.yaml`**](./configs/experiment/shapenet/transformer.yaml):

```yaml
# @package _global_
defaults:
  - override /model: transformer  
  - override /tracker: development_tracker
  - override /optimizer: lion
  
name: shapenet-car-transformer-float16

trainer:
  precision: float16
```

**Breaking down the experiment config:**

- **`override /model: transformer`**: Use `configs/model/transformer.yaml` instead of the placeholder `???` in the base config
- **`override /tracker: development_tracker`**: Select the W&B tracker configuration
- **`override /optimizer: lion`**: Override the default AdamW optimizer with Lion
- **`trainer.precision: float16`**: Override the trainer's default `float32` precision

The `override` keyword ensures the experiment's choice takes precedence over any defaults, preventing accidental config merging issues.

**Creating new experiments:**

To run a different model on the same dataset:

1. Create a new experiment file (e.g., `configs/experiments/shapenet/my_model.yaml`)
2. Specify the model config to use
3. Add any model-specific overrides
4. Keep tracker and other settings as needed

### Running experiments

**Basic execution:**

To train a model with a specific configuration (from the root of the repository):

```bash
uv run noether-train --hp tutorial/configs/train_shapenet.yaml +experiment/shapenet=transformer tracker=disabled trainer.max_epochs=10
```

```bash
uv run noether-train --hp tutorial/configs/train_shapenet.yaml +experiment/shapenet=ab_upt tracker=disabled trainer.max_epochs=10
```

To enable experiment tracking, simply remove the `tracker=disabled` override:

```bash
uv run noether-train --hp tutorial/configs/train_shapenet.yaml +experiment/shapenet=transformer
```


> [!WARNING]
> Make sure to either set `dataset_root` in `train_shapenet.yaml` or add it to the command line via `dataset_root=\"<path to dataset root>\"

You'll need to configure your W&B API key on first run and update [**`configs/tracker/development_tracker.yaml`**](./configs/tracker/development_tracker.yaml) with your project details.

**Single parameter overrides:**

```bash
uv run noether-train --hp tutorial/configs/train_shapenet.yaml \
  +experiment/shapenet=transformer \
  trainer.max_epochs=100
```

**Multiple parameter overrides:**

To modify multiple related parameters (e.g., changing Transformer dimensions):

```bash
uv run noether-train --hp tutorial/configs/train_shapenet.yaml \
  +experiment/shapenet=transformer \
  model.hidden_dim=256 \
  model.num_heads=4 \
```
Note: When changing `hidden_dim`, ensure `num_heads` divides it evenly (i.e., `hidden_dim % num_heads == 0`).


### Pydantic schemas for type safety

While Hydra handles configuration composition, **Pydantic schemas** provide runtime validation and type safety. 
Every class in the `Noether Framework` has a corresponding Pydantic schema that validates configuration: Checks types, ranges, and constraints before training begins.

**Schema hierarchy:**

All schemas in the `Noether Framework` follow an inheritance pattern. For example, model schemas inherit from `ModelBaseConfig`:

```python
class ModelBaseConfig(BaseModel):
    kind: str
    """Kind of model to use, i.e. class path"""
    
    name: str
    """Name of the model. Needs to be unique"""
    
    optimizer_config: OptimizerConfig | None = None
    """The optimizer configuration to use for training."""
    
    initializers: list[AnyInitializer] | None = Field(None)
    """List of initializers configs to use for the model."""
    
    is_frozen: bool | None = False
    """Whether to freeze the model parameters."""
    
    forward_properties: list[str] | None = []
    """List of properties to be used as inputs for the forward pass."""

    model_config = {"extra": "forbid"}
```

The `extra: "forbid"` setting ensures that typos in YAML files are caught immediately, preventing silent configuration errors.

### Example: Transformer configuration schema

All models in `Noether` use schema composition and validation. The schema hierarchy for the Transformer models looks like:

```
ModelBaseConfig (base for all models)
    └── TransformerConfig (Transformer-specific config)
          └── TransformerBlockConfig (component config)
```

**TransformerBlockConfig** defines individual block parameters:

```python
class TransformerBlockConfig(BaseModel):
    """Configuration for a transformer block."""

    hidden_dim: int = Field(..., ge=1)
    """Hidden dimension of the transformer block."""

    num_heads: int = Field(..., ge=1)
    """Number of attention heads."""

    mlp_hidden_dim: int | None = Field(None)
    """Hidden dimension of the MLP layer."""

    mlp_expansion_factor: int | None = Field(None, ge=1)
    """Expansion factor for MLP hidden dimension."""

    drop_path: float = Field(0.0, ge=0.0, le=1.0)
    """Stochastic depth probability."""

    attention_constructor: Literal[
        "dot_product",
        "perceiver", 
        "transolver",
        "transolver_plusplus",
    ] = "dot_product"
    """Type of attention mechanism to use."""

    use_rope: bool = Field(False)
    """Whether to use Rotary Positional Embeddings."""

    # ... additional fields omitted for brevity

    @model_validator(mode="after")
    def set_mlp_hidden_dim(self):
        if self.mlp_hidden_dim is None:
            if self.mlp_expansion_factor is None:
                raise ValueError(
                    "Either 'mlp_hidden_dim' or 'mlp_expansion_factor' must be provided."
                )
            self.mlp_hidden_dim = self.hidden_dim * self.mlp_expansion_factor
        return self
```

**TransformerConfig** extends the block config:

```python
class TransformerConfig(TransformerBlockConfig, ModelBaseConfig):
    """Configuration for a Transformer model."""

    model_config = ConfigDict(extra="forbid")

    depth: int
    """Number of transformer blocks in the model."""
    
    mlp_expansion_factor: int = 4
    """Override default: expansion factor for MLP hidden dimension."""

    @model_validator(mode="after")
    def set_mlp_hidden_dim(self):
        if self.mlp_hidden_dim is None:
            if self.mlp_expansion_factor is None:
                raise ValueError(
                    "Either 'mlp_hidden_dim' or 'mlp_expansion_factor' must be provided."
                )
            self.mlp_hidden_dim = self.hidden_dim * self.mlp_expansion_factor
        return self
```

**Multiple inheritance** means `TransformerConfig` inherits:
- Model management from `ModelBaseConfig` (optimizer, freezing, etc.)
- Block parameters from `TransformerBlockConfig` (attention, MLP, etc.)
- Adds Transformer model parameters (`depth`)
- Overrides defaults (sets `mlp_expansion_factor = 4`)

### From schema to YAML

Understanding the schema tells you which YAML fields are required and optional. For a minimal Transformer config:

```yaml
kind: tutorial.model.Transformer
name: transformer
hidden_dim: 192
depth: 12
num_heads: 3
optimizer_config: ${optimizer}
```

### Tutorial configuration schema

Your downstream project must define a top-level configuration schema that specifies the complete experiment structure. 
For this tutorial, the schema is:

```python
class TutorialConfigSchema(ConfigSchema):
    data_specs: AeroDataSpecs
    model: AnyModelConfig = Field(..., discriminator="name")
    trainer: AutomotiveAerodynamicsCfdTrainerConfig
    datasets: dict[str, AeroDatasetConfig]
    dataset_statistics: AeroStatsSchema | None = None
```

- **Inherits from `ConfigSchema`**: The base configuration schema from `Noether Framework`
- **`data_specs`**: Defines the data structure (field names, dimensions, types) for aerodynamics tasks
- **`model`**: Union type using discriminator pattern - accepts any model config that will defined 
- **`trainer`**: Specifies the trainer configuration (specific to automotive aerodynamics CFD)
- **`datasets`**: Dictionary of dataset configurations
- **`dataset_statistics`**: Optional normalization statistics for the dataset

`AnyModelConfig` is a union of all model configs we define in this project. 
Pydantic uses the `name` of the configured model as discriminator to select the correct schema.
It tells Pydantic to use the `name` field to determine which specific model schema to validate against:


```python
AnyModelConfig = Union[
    TransformerConfig,
    TransolverConfig,
    UPTConfig,
    ABUPTConfig,
    TransolverPlusPlusConfig,
    CompositeTransformerConfig,
]

```

**Where schemas are defined:**

All tutorial schemas live in the [`schemas/`](./schemas/) directory:

```
schemas/
├── callbacks/          # Callback configuration schemas
├── datasets/           # Dataset configuration schemas
├── models/             # Model configuration schemas
├── pipelines/          # Pipeline configuration schemas
└── trainers/           # Trainer configuration schemas
```

Each module in your project should have a corresponding schema that defines its configuration interface.

The `kind` field in most configs specifies the class path for instantiation. The Factory pattern uses this to dynamically import and instantiate the correct class with the validated configuration.

## Object instantiation

The objects we instantiate in the `Noether Framework` via configs are instantiated via a factory.
To do this, the config of the object needs to contain a `kind`, i.e., the class path of the class.
The remaining variables are passed to the constructor of the class via the config object we created after the pydantic schema evaluation.
An example is given above in the Transformer config, where `kind: tutorial.model.Transformer` indicates which model class to instantiate.

## The Dataset ([`datasets/`](../src/noether/data/datasets/cfd/shapenet_car/dataset.py))

The `Dataset` class serves as the bridge between raw (or preprocessed) data stored on disk and the **multi-stage pipeline** that transforms individual samples into batches for model training (which we discuss in the next section). 
It defines how to load and access individual data tensors for each sample.

**The Dataset enables:**
- Load individual data samples from disk
- Provide tensor-level data access through modular methods
- Apply per-tensor normalization and transformations
- Support flexible data loading for different model requirements

This tutorial uses the pre-implemented [**`ShapeNetCarDataset`**](../src/noether/data/datasets/cfd/shapenet_car/dataset.py) from the `Noether` package.

**Dataset class hierarchy:**

```
torch.utils.data.Dataset (PyTorch base)
    └── noether.data.Dataset (Noether base with getitem_* pattern)
          └── noether.data.datasets.cfd.AeroDataset (CFD aerodynamics API)
                └── ShapeNetCarDataset (ShapeNet-Car implementation)
```

The [`AeroDataset`](../src/noether/data/datasets/cfd/dataset.py) provides a general API for CFD aerodynamics datasets (AhmedML, DrivAerML, DrivAerNet++, ShapeNet-Car, etc.), ensuring consistent interfaces across different automotive aerodynamics datasets.

### The `getitem_*` pattern: Modular data loading

Traditional PyTorch datasets use a single `__getitem__` method to load all data for a sample. This approach has several limitations:
- Becomes complex when different models need different inputs from the same dataset
- Difficult to selectively load subsets of data
- Hard to maintain when adding new data fields
- Forces loading unused data for some experiments

The `Noether Framework` uses a **modular `getitem_*` pattern** where each data tensor has its own dedicated loading method. This enables:
- **Modularity**: Each method loads one specific tensor
- **Flexibility**: Selectively load only required tensors via configuration
- **Maintainability**: Easy to add new data fields without modifying existing code
- **Clarity**: Self-documenting through method names (e.g., `getitem_surface_pressure`)

**Example implementation:**

```python
def _load(self, idx: int, filename: str) -> torch.Tensor:
    """
    Loads a tensor from a file within a specific sample directory.

    Args:
        idx: Index of the sample to load.
        filename: Name of the file to load from the sample directory.

    Returns:
        The loaded tensor.
    """
    # Use modulo to handle dataset repetitions
    idx = idx % len(self.uris)
    sample_uri = self.uris[idx] / filename
    return torch.load(sample_uri, weights_only=True)

def getitem_surface_position(self, idx: int) -> torch.Tensor:
    """Retrieves surface position coordinates (num_surface_points, 3)."""
    return self._load(idx=idx, filename="surface_points.pt")

def getitem_surface_pressure(self, idx: int) -> torch.Tensor:
    """Retrieves surface pressure values (num_surface_points, 1)."""
    return self._load(idx=idx, filename="surface_pressure.pt").unsqueeze(1)
```

**Design pattern:**
- **Helper methods** (e.g., `_load`) keep code DRY and handle common operations
- **Descriptive names** make it clear what each method loads
- **Consistent signature**: All `getitem_*` methods take `idx` and return a tensor
- **Tensor-level operations**: Shape transformations (e.g., `unsqueeze`) applied immediately

### ShapeNet-Car dataset structure

The ShapeNet-Car dataset contains CFD simulation data for 889 car geometries, with each data point consisting of preprocessed PyTorch tensors stored on disk. 

> [!NOTE]
> To download and preprocess the data, see the [ShapeNet-Car dataset README](../src/noether/data/datasets/cfd/shapenet_car/README.MD).

**Available data tensors:**

Each simulation provides the following fields through corresponding `getitem_*` methods:

| Tensor | Method | Shape | Description |
|--------|--------|-------|-------------|
| **Surface Position** | `getitem_surface_position` | `(N_surf, 3)` | 3D coordinates of surface mesh points |
| **Surface Pressure** | `getitem_surface_pressure` | `(N_surf, 1)` | Pressure values at surface points |
| **Surface Normals** | `getitem_surface_normals` | `(N_surf, 3)` | Normal vectors at surface points |
| **Volume Position** | `getitem_volume_position` | `(N_vol, 3)` | 3D coordinates of volume mesh points |
| **Volume Velocity** | `getitem_volume_velocity` | `(N_vol, 3)` | Velocity vectors at volume points |
| **Volume Normals** | `getitem_volume_normals` | `(N_vol, 3)` | Normal vectors (pointing to nearest surface) |
| **Volume SDF** | `getitem_volume_sdf` | `(N_vol, 1)` | Signed Distance Function to nearest surface |

**Note on surface SDF:**
There is no `getitem_surface_sdf` method because surface SDF values are always zero (points on the surface have zero distance to the surface). This constant tensor is created automatically in the multi-stage pipeline when needed, avoiding redundant disk storage.

### Dataset configuration

Datasets in `Noether` are instantiated by the [`DatasetFactory`](../src/noether/core/factory/dataset.py), which uses configuration files to create dataset instances with appropriate settings.

**Basic dataset configuration structure:**

The [**`configs/datasets/shapenet_dataset.yaml`**](./configs/datasets/shapenet_dataset.yaml) file defines dataset configurations for different splits:

```yaml
train:
  root: ${dataset_root} 
  kind: ${dataset_kind}
  split: train
  pipeline: ${pipeline}
  dataset_normalizers: ${dataset_normalizers}
  excluded_properties: ${excluded_properties}

test:
  root: ${dataset_root}
  kind: ${dataset_kind}
  split: test
  pipeline: ${pipeline}
  dataset_normalizers: ${dataset_normalizers}
  excluded_properties: ${excluded_properties}
```

**Configuration parameters:**

- **`root`**: Path to the dataset directory on disk
- **`kind`**: Full class path to the dataset class (e.g., `noether.data.datasets.cfd.ShapeNetCarDataset`)
- **`split`**: Data split identifier (`train`, `test`, `val`, etc.) used by the dataset to select appropriate samples
- **`pipeline`**: Reference to the multi-stage pipeline configuration
- **`dataset_normalizers`**: Reference to tensor normalization configurations
- **`excluded_properties`**: List of `getitem_*` methods to skip during data loading

**Advanced: Multiple dataset configurations**

You can define multiple dataset configurations for different evaluation scenarios:

```yaml
test_repeat:
  root: ${dataset_root}
  kind: ${dataset_kind}
  split: test
  pipeline: ${pipeline}
  dataset_normalizers: ${dataset_normalizers}
  excluded_properties: ${excluded_properties}
  dataset_wrappers:
    - kind: noether.data.base.wrappers.RepeatWrapper
      repetitions: 10
```

**Dataset wrappers:**

The `RepeatWrapper` loops over the dataset multiple times (10× in this example) to reduce variance during evaluation. Other useful wrappers [include](../src/noether/data/base/wrappers/):
- **`SubsetWrapper`**: Select specific indices from the dataset
- **`ShuffleWrapper`**: Randomize sample order

This flexibility allows you to:
- Use different pipelines for train vs. test datasets
- Create multiple evaluation sets with different sampling strategies
- Apply different normalizations to different splits 

### Selective data loading with `excluded_properties`

By default, all `getitem_*` methods are called when loading a sample. 
However, different models often require different input tensors. 
The `excluded_properties` configuration allows selective loading:

```yaml
# Example: Exclude normal vectors for a model that doesn't use them
excluded_properties: 
  - surface_normals
  - volume_normals
```
A point-based Transformer might only need positions and surface pressure and volume velocity:
```yaml
# Load only essential tensors
excluded_properties:
  - surface_normals
  - volume_normals
  - volume_sdf
```
Now those additional features are excluded from data loading, while a more complex model uses all available features:
```yaml
# Load everything
excluded_properties: []
```

This pattern enables using the same dataset class for different models without modifying code.

### Essential dataset methods

Beyond the `getitem_*` methods, dataset classes implement standard PyTorch dataset methods:

**`__len__` method:**

Defines the total number of samples for one epoch:

```python
def __len__(self) -> int:
    """Returns the total size of the dataset."""
    return len(self.uris) * self.num_repeats
```

This calculation accounts for dataset repetitions, useful for oversampling small datasets during training.

**Additional methods:**

Most other methods follow standard PyTorch `Dataset` patterns. If you're unfamiliar with PyTorch datasets, review the [official PyTorch dataset tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).

### Tensor normalization with decorators

In the `Noethern Framework`, most of the normalization happens at the **tensor level** immediately after loading, using a decorator pattern for clean, declarative code.

**The `@with_normalizers` decorator:**

Apply normalization to any `getitem_*` method by adding a decorator:

```python
@with_normalizers("surface_position")
def getitem_surface_position(self, idx: int) -> torch.Tensor:
    """Retrieves surface positions (num_surface_points, 3)"""
    return self._load(idx=idx, filename=self.filemap.surface_position)
```

**How it works:**

1. The decorator identifies which normalizer(s) to apply using the key (`"surface_position"`)
2. Looks up the normalizer configuration in the dataset's `dataset_normalizers` config
3. Applies the normalization transformation to the loaded tensor
4. Returns the normalized tensor

**Configuring normalizers:**

All normalizers are defined in [`noether.data.preprocessors.normalizers`](../src/noether/data/preprocessors/normalizers.py).

```yaml
surface_pressure:
  - kind: noether.data.preprocessors.normalizers.MeanStdNormalization
    mean: ${dataset_statistics.surface_pressure_mean}
    std: ${dataset_statistics.surface_pressure_std}

surface_position:
  - kind: noether.data.preprocessors.normalizers.MeanStdNormalization
    mean: ${dataset_statistics.surface_position_mean}
    std: ${dataset_statistics.surface_position_std}
```
We now defined the `surface_pressure`, which maps to a `MeanStdNormalization`, using a configurable mean and std. 

**Composing multiple normalizers:**

Note that each key configures a **list** of normalizers, allowing you to compose a chain of normalization methods.
All normalization preprocessor must be invertible so that we can denormalize the data for evaluation.
Each normalizer is wrapped by the `noether.data.preprocessor.ComposePreProcess`, which can contain multiple preprocessors applied sequentially.
Each `noether.data.preprocessor.PreProcessor` must implement the **`denormalize`** method.
The **`ComposePreProcess.inverse`** method calls the **`denormalize`** method of all normalizers in the `ComposePreProcess` in reverse order, ensuring that data normalization can be inverted correctly.

### Computing dataset statistics

To use normalizers like `MeanStdNormalization`, you need to compute statistics from your training data.

**Step 1: Compute statistics**

Run the statistics calculation tool:

```bash
python src/noether/data/tools/calculate_statistics.py \
  --dataset_kind=noether.data.datasets.cfd.ShapeNetCarDataset \
  --root=/path/to/shapenet_car/ \
  --split=train \
  --exclude_attributes=volume_velocity,volume_pressure,volume_vorticity,surface_normals,surface_friction
```

**Parameters explained:**
- **`--dataset_kind`**: Full class path to your dataset
- **`--root`**: Path to dataset directory
- **`--split`**: Which split to compute statistics from (typically `train`)
- **`--exclude_attributes`**: Properties to skip (either unavailable or not used)

> [!NOTE]
> We exclude certain properties because they're not available in ShapeNet-Car, even though the general `AeroDataset` interface defines `getitem_*` methods for them.


The statistics need to be manually added to a YAML file in [**`configs/dataset_statistics/`**](./configs/dataset_statistics/):

### Noether dataset zoo

The `Noether Framework` includes pre-implemented datasets for CFD aerodynamics in [`noether.data.datasets.cfd`](../src/noether/data/datasets/cfd/):

| Dataset | Class Path |
|---------|------------|
| **ShapeNet-Car** | [noether.data.datasets.cfd.ShapeNetCarDataset](../src/noether/data/datasets/cfd/shapenet_car/dataset.py) |
| **AhmedML** | [noether.data.datasets.cfd.AhmedMLDataset](../src/noether/data/datasets/cfd/caeml/ahmedml/dataset.py) |
| **DrivAerML** | [noether.data.datasets.cfd.DrivAerMLDataset](../src/noether/data/datasets/cfd/caeml/drivaerml/dataset.py) |
| **DrivAerNet++** | [noether.data.datasets.cfd.DrivAerNetDataset](../src/noether/data/datasets/cfd/drivaernet/dataset.py) |
| **Wing Dataset** | [noether.data.datasets.cfd.EmmiWingDataset](../src/noether/data/datasets/cfd/caeml/ahmedml/dataset.py) |

All datasets share the `AeroDataset` interface, ensuring consistent access patterns and easy switching between datasets.

**Creating custom datasets:**

To implement a custom dataset:

1. Inherit from `noether.data.Dataset` (or `noether.data.datasets.cfd.AeroDataset`)
2. Implement required `getitem_*` methods for your data fields
3. Override `__init__` to discover and filter your data samples
4. Add `@with_normalizers` decorators where normalization is needed
5. Create corresponding Pydantic schema in your `schemas/datasets/` directory
6. Configure the normalizers 

See the [boilerplate project](../boilerplate_project/datasets/) for a minimal dataset implementation example.

## The Multi-Stage Pipeline ([`pipeline/`](./pipeline/))

The multi-stage pipeline serves as the interface between the dataset class and the model/trainer (which we discuss later).
It defines how to combine individual samples from the dataset into batches that are fed to the model.
Each batch contains the model inputs for the forward pass and the corresponding targets needed to compute the loss.

The multi-stage pipeline has three sequential stages:

1. **Sample processor pipeline**: Sample processors act on individual data samples (i.e., data points).
2. **Collation**: The collator pipeline collates individual samples into a batch.
3. **Batch processor pipeline**: Batch processors act on the entire batch.

This sequential processing gives the **multi-stage pipeline** its name.
In this project, most of the computation occurs during the sample processing stage.

A basic implementation of a custom [MultiStagePipeline](../src/noether/data/pipeline/multistage.py) looks like this:

```python
from noether.data.pipeline import MultiStagePipeline

class CustomMultiStagePipeline(MultiStagePipeline):
  def __init__(self, **kwargs):
    super().__init__(
      preprocessors=[],
      collators=[],
      postprocessors=[],
      **kwargs,
    )
```

You need to provide three lists to the multi-stage pipeline (which are all empty in the example above): one for sample processors, one for collators, and one for batch processors.
The `MultiStagePipeline` iterates through each list sequentially.
The output from one processor becomes the input for the next, making the order of operations **crucial** for all three stages.

### Sample processors

To understand the [AeroMultistagePipeline](./pipeline/multistage_pipelines/aero_multistage.py), it's essential to understand the data processing flow for this project.

We're dealing with CFD aerodynamic simulations that have both a surface and a volume mesh/field.
Each point in these fields has three coordinates `(x, y, z)`, one or more target values (e.g., `pressure`, `velocity`,  `vorticity`, `wallshear stress`, etc.), and potentially additional features (e.g., `SDF`, surface/volume normals).
The target values and features can vary depending on whether the point belongs to the surface or volume and which dataset is used.
From now on, we'll refer to these additional features as **physics features**.
We do not consider global features for this project.
The data structure for our tasks is defined in, for example, [**`configs/data_specs/shapenet_car.yaml`**](./configs/data_specs/shapenet_car.yaml), which corresponds to the [**`AeroDataSpecs`**](../src/noether/core/schemas/dataset.py#186) schema.

The models we use can be roughly **divided** into two classes:
1. **Point-based models**, where the input points to the model's encoder are also the points used for predicting the output values (e.g., `Transformer`, `Transolver`).
2. **Query-based models**, which use additional query points (distinct from the input points) for predicting output values (e.g., `UPT`, `AB-UPT`).

This means we have to build a multi-stage pipeline that works for both point-based and query-based models.

We will now outline the sample processor pipeline required for these models:

1. Some input tensors have **constant** values. For example, the `SDF` for the surface mesh is always zero (as discussed earlier).
   Therefore, we first create **default tensors** if needed.
   Because this step occurs before batch collation, it's considered a **sample processing** step.
2. Next, we subsample the entire simulation mesh to a specified number of surface and volume points and, if used, query points.
   For both input and query points, we define how many to sample from the surface and how many from the volume. If we train AB-UPT, we sample anchor points instead of input/query points.
3. If we use query points, their corresponding physical quantities become the model's prediction targets.
   If we only use input points, their values are the output targets (labels).
   Hence, we need to rename the relevant values to `targets` based on whether the model uses input points or query points for its predictions.

The high-level pipeline is visualized in the image below:

![Pipeline data flow](./images/shapenet_car_models_data_flow.png)

This entire pipeline is implemented in the [**`_build_sample_processor_pipeline`**](./pipeline/multistage_pipelines/aero_multistage.py#176) method in the `AeroMultistagePipeline` class, which composes the list of sample processors classes based on the three steps listed above.
Please have a look at the code to understand what it is doing.
This method returns a list of individual [**`SampleProcessor`**](../src/noether/data/pipeline/sample_processor.py) instances.
Each sample processor takes a sample as input (which is a dictionary with the resul of all the `getitem_*` methods called by the dataset for one data point) and does some form of processing on one or more tensors of the sample.
Note that the order is important, as the sample processors are called sequentially.

When the multi-stage pipeline runs, the sample processors are called as [follows](../src/noether/data/pipeline/multistage.py#100): 

```python
# pre-process on a sample level
samples = [deepcopy(sample) for sample in samples]  # copy to avoid changing method input
for sample_processor in self.sample_processors:
  for idx, sample in enumerate(samples):
    # sample = {'surface_pressure: torch. Tensor[...], 'surface_position': torch.Tensor[...], .....}
    # each key in the sample is the output of a getitem_* method of the dataset 
    samples[idx] = sample_processor(sample)
```

Each **sample processor** takes a sample as input and returns the (pre)processed sample. 
As mentioned, the order is **crucial**.
Each `SampleProcessor` must implement the `__call__(self, sample: dict[str, Any]) -> dict[str, Any]` method.
This method receives a dictionary containing the sample's tensors as input. 
The `SampleProcessor`'s goal is to apply a specific processing step to the corresponding values for one or more keys in the sample dictionary.
See individual sample processor implementations (e.g., [**`PointSampling`**](../src/noether/data/pipeline/sample_processors/point_sampling.py)) for detailed examples.

### Collators

The code for calling the collators in the multi-stage pipeline looks as [follows](../src/noether/data/pipeline/multistage.py#107):

```python
batch = {}
for batch_collator in self.collators:
  sub_batch = batch_collator(samples)
  # make sure that there is no overlap between collators
  for key, value in sub_batch.items():
    if key in batch:
      raise ValueError(f"Key '{key}' already exists in batch. Collators must not overlap in keys.")
    batch[key] = value
```

Each collator defines how to merge certain keys from each sample into a batch.
In most cases, the [**`DefaultCollator`**](../src/noether/data/pipeline/collators/default.py), where tensors are simply concatenated along the batch dimension, will suffice.
However, when creating [sparse tensors](../src/noether/data/pipeline/sample_processors/sparse_tensor_offset.py), for example, a more sophisticated collation approach is required.
We define the collator pipeline in the [**`_build_collator_pipeline`**](./pipeline/multistage_pipelines/aero_multistage.py#94) method.
Only when dealing with supernodes do we require additional collator classes such as the custom-implemented `SparseTensorOffsetCollator` (e.g., for AB-UPT and UPT).

### Batch processors

In this project, we do not use any batch processors.
Nevertheless, they work in the same way as sample processors.
However, instead of processing individual samples, they process the collated batch.
Below is the code showing how batch processors are [called](../src/noether/data/pipeline/multistage.py#116): 

```python 
# process the batch
for batch_processor in self.batch_processors:
    batch = batch_processor(batch)
```

## The Trainer ([`trainers/`](./trainers/))

The [`AutomotiveAerodynamicsCFDTrainer`](./trainers/automotive_aerodynamics_cfd.py) is a specialized trainer designed for automotive Computational Fluid Dynamics (CFD) tasks, specifically for the AhmedML, DrivAerML, DrivAerNet++, ShapeNet-Car, and Emmi-Wing datasets.
Its primary role is to manage the training step by processing model outputs, computing a flexible weighted loss, and returning the results.

### BaseTrainer implementation

To implement a custom `Trainer` for a downstream project, you must extend the [noether.training.trainers.BaseTrainer](../src/noether/training/trainers/base.py) class. 
The `BaseTrainer` handles the full training loop and provides the following two key methods:

```python

def loss_compute(
        self, forward_output: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> LossResult | tuple[LossResult, dict[str, torch.Tensor]]:
        """
        Each trainer that extends this class needs to implement a custom loss computation using the targets and the model output.

        Args:
            forward_output: Output of the model after the forward pass.
            targets: Dict with target tensors needed to compute the loss for this trainer.

        Returns:
            A dict with the (weighted) sub-losses to log.
        """
        raise NotImplementedError("Subclasses must implement loss_compute.")


def train_step(self, batch: dict[str, Tensor], model: torch.nn.Module) -> TrainerResult:
        """Overriding this function is optional. By default, the `train_step` of the model will be called and is
        expected to return a TrainerResult. Trainers can override this method to implement custom training logic.
        
        Args:
            batch: Batch of data from which the loss is calculated.
            model: Model to use for processing the data.
            
        Returns:
            TrainerResult dataclass with the loss for backpropagation, (optionally) individual losses if multiple 
            losses are used, and (optionally) additional information about the model forward pass that is passed 
            to the callbacks (e.g., the logits and targets to calculate a training accuracy in a callback).
        """
        forward_batch, targets_batch = self._split_batch(batch)
        forward_output = model(**forward_batch)
        additional_outputs = None
        losses = self.loss_compute(forward_output=forward_output, targets=targets_batch)

        if isinstance(losses, tuple) and len(losses) == 2:
            losses, additional_outputs = losses

        if isinstance(losses, torch.Tensor):
            return TrainerResult(total_loss=losses, additional_outputs=additional_outputs, losses_to_log={'loss': losses})
        elif isinstance(losses, list):
            losses = {f"loss_{i}": loss for i, loss in enumerate(losses)}

        if len(losses) == 0:
            raise ValueError("No losses computed, check your output keys and loss function.")

        return TrainerResult(
            total_loss=sum(losses.values(), start=torch.zeros_like(next(iter(losses.values())))),
            losses_to_log=losses,
            additional_outputs=additional_outputs,
        )
```

**Understanding the two key methods:**

As an end-user, you need to implement the methods: **`loss_compute`** and sometimes **`train_step`**.

The **`train_step`** method receives the batch from the multi-stage pipeline and the **`model`** being trained (which can be a `DistributedDataParallel` model when training on multiple GPUs).

In the base implementation, the batch is split into two sub-batches:

1. **Forward batch**: Contains all tensors needed for the forward pass. The model receives the `forward_batch` as named keyword arguments, and the forward pass is computed.
2. **Targets batch**: Contains tensors needed for loss computation. The **`loss_compute`** method computes the custom loss for your task.

For task-specific implementations, see the [`AutomotiveAerodynamicsCFDTrainer`](./trainers/automotive_aerodynamics_cfd_trainer.py) example.

> [!IMPORTANT]
> We give a warming if there are keys in the batch that do not end in either the `forward batch` or the `target batch`. This means that the collator returns tensors that are not used during the forward pass. 

**Return value requirements:**

The **`train_step`** method must always return the **`TrainerResult`** dataclass, which should contain:
- A scalar value of the total loss used to compute gradients (can be a weighted sum of multiple losses)
- A dictionary with the losses you want to log
- Optionally, a dictionary with additional output for logging

**When to override `train_step`:**

The **`train_step`** method defined in the **`BaseTrainer`** class fits most general deep learning forward passes. 
However, you can decide whether this implementation is sufficient for your downstream training task.
If not, you can always implement a custom **`train_step`** method in the child trainer class (as has been done in the [boiler plate project trainer](../boilerplate_project/trainer/base.py)).

### BaseTrainer configuration

When using the default **`train_step`** method, you must define both the **`forward_properties`** and the **`target_properties`** to define which tensors are part of the `forward_batch` and which tensors are part of the `target_batch`.

In this tutorial, the target properties are fixed per dataset, while the **`forward_properties`** depend on the model.
Therefore, we define them as follows:

```yaml
target_properties:
  - surface_pressure_target
  - volume_velocity_target
forward_properties: ${model.forward_properties}
```

**Required BaseTrainer parameters:**

The following parameters must be defined for the **`BaseTrainer`**:

```yaml
kind: tutorial.trainers.AutomotiveAerodynamicsCFDTrainer # which trainer to load 
max_epochs: 500
effective_batch_size: 1 
log_every_n_epochs: 1 # optional but best practice to define
callbacks: ${callbacks} # which callbacks to run
```

### AutomotiveAerodynamicsCFDTrainer implementation

The most important variables in the **`__init__`** method are the loss weights, which give you fine-grained control over the training objective.

**Loss weight hierarchy:**

The loss has two levels of weights:

- **Individual weights**: Parameters like `surface_pressure_weight` and `volume_velocity_weight` control the importance of a specific physical quantity in the total loss.
- **Group weights**: The `surface_weight` and `volume_weight` parameters apply an additional weight to all surface-related or volume-related losses, respectively.

During initialization, the trainer uses these weights to build an internal **`loss_items`** list.
The **`output_modes`** parameter (e.g., `['surface_pressure', 'volume_velocity']`) specifies which of these potential losses should be computed during training.

**Custom loss calculation (`loss_compute`):**

This method contains the core logic of the trainer for computing the loss.
It calculates the final loss by iterating through the **`loss_items`** configured during initialization.
For each item (like `surface_pressure`), it first checks that its weight is non-zero and that the model produced a corresponding output key.
This flexible system allows you to easily experiment with different combinations of output objectives without changing the underlying code.
When using only a single loss value, the **`loss_compute`** method is not needed and can be implemented directly inside the forward function (by overriding the base **`train_step`** method, as done in the [boilerplate project](../boilerplate_project/trainer/base.py)).

## Models (`models/`)

Building models in the `Noether Framework` is straightforward and follows the same patterns as standard PyTorch models that inherit from `torch.nn.Module`.

To be compatible with the `Noether Trainer`, all models must inherit from [noether.core.models.Model](../src/noether/core/models/single.py) (or [CompositeModel](../src/noether/core/models/composite.py) for multi-component architectures, discussed later).
Beyond this, a model is implemented just like any PyTorch model: define layers in the constructor (`__init__`) and implement the `forward` method.

### The `ModelBaseConfig` schema

Each model in the `Noether Framework` must inherit from the [noether.core.models.Model](../src/noether/core/models/single.py) class.
The config schema for models is defined by `ModelBaseConfig`:

```python
class ModelBaseConfig(BaseModel):
    kind: str
    """Kind of model to use, i.e. class path"""
    name: str
    """Name of the model. Needs to be unique"""
    optimizer_config: OptimizerConfig | None = None
    """The optimizer configuration to use for training the model. When a model is used for inference only, this can be left as None."""
    initializers: list[AnyInitializer] | None = Field(None)
    """List of initializers configs to use for the model."""
    is_frozen: bool | None = False
    """Whether to freeze the model parameters (i.e., not trainable)."""
    forward_properties: list[str] | None = []
    """List of properties to be used as inputs for the forward pass of the model. Only relevant when the train_step of the BaseTrainer is used. When overridden in a class method, this property is ignored."""
```

**Key configuration parameters:**

- **`kind`**: The full class path to the model class (e.g., `tutorial.model.Transformer`).
- **`name`**: A unique identifier for the model, typically overridden in child config classes to match with the correct model configuration.
- **`optimizer_config`**: The optimizer configuration for training. Can be `None` when loading a model for inference only.
- **`initializers`**: Optional list of initializer configs for loading pre-trained weights or custom weight initialization.
- **`is_frozen`**: Boolean flag to freeze all model parameters (useful for transfer learning or ensemble models).
- **`forward_properties`**: List of properties to be used as inputs for the model's forward pass. Only relevant when using the `BaseTrainer`'s default `train_step` method.

> [!NOTE]
> In the `Noether Framework`, optimizers are attached to models rather than being global. This design allows different components of composite models to use different optimizers and learning rates.

### Implementing a custom model

A minimal custom model implementation looks as follows:
 
**Python implementation dummy code**:
 
```python
from noether.core.models import Model


class CustomModel(Model):
    def __init__(self, model_config: CustomModelConfig, **kwargs):
        # the model config needs to be passed to the parent Model class 
        super().__init__(model_config=model_config, **kwargs)
    
        self.config = model_config
        
        # Define your model layers here
        self.encoder = torch.nn.Linear(model_config.input_dim, model_config.hidden_dim)
        self.decoder = torch.nn.Linear(model_config.hidden_dim, model_config.output_dim)
    
    def forward(self, input_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_tensor: torch tensor with data 
        
        Returns:
            Dictionary containing model outputs.
        """
        # Example: extract inputs from batch
        x = input_tensor
        
        # Forward pass
        hidden = self.encoder(x)
        output = self.decoder(hidden)
        
        return {'output':output}
```

```yaml
kind: path.to.CustomModel
name: custom_model
input_dim: 3
hidden_dim: 128
output_dim: 1
optimizer_config: ${optimizer}  # Reference to optimizer defined elsewhere
forward_properties:
  - input_tensor
```

### The `BaseModel` for standardization

To unify input representation, output structure, and input conditioning across all baseline models in this tutorial, we provide a [BaseModel](./model/base.py) class.
This `BaseModel` inherits from `noether.core.models.Model` and contains common utilities that can be reused across different model architectures:

1. **Surface and volume bias projection**: An MLP projection layer to handle domain-specific biases.
2. **Physics feature projection**: A linear layer to map physics features (e.g., SDF, normals) to the model's hidden dimension.
3. **Positional embeddings**: Sine-cosine or linear positional embedding layers for input coordinates.
4. **Output projection**: A final linear layer to project from the hidden dimension to the number of predicted physical quantities.

**Benefits of using `BaseModel` in a downstream project:**
- Reduces code duplication across different model implementations
- Ensures consistent input/output interfaces
- Simplifies distinguishing between surface and volume mesh coordinates
- Provides standardized feature processing

### Handling model outputs

Physical quantities predicted for surface points often differ from those for volume points. For example:
- **Surface predictions**: pressure, wall shear stress
- **Volume predictions**: velocity, pressure, vorticity

The [gather_outputs](./model/base.py#174) method in the `BaseModel` class handles this heterogeneity:
- Takes the entire output tensor and a surface mask
- Splits the output tensor to isolate surface predictions from volume predictions
- Returns a structured dictionary that maps to physical quantities

**Example output structure:**
```python
{
    'surface_pressure': tensor[...],      # dimension 0 of surface outputs
    'volume_velocity': tensor[...],       # dimensions 1:4 of volume outputs
    'surface_friction': tensor[...],       # dimension 4:7 of surace outputs
    ....
}
```

By using `gather_outputs` consistently across all models, the output dictionary is structured in a way that the trainer's `loss_compute` method can process uniformly. This design allows the same trainer to work with all model architectures without modification.

### Composite Models

A **composite model** consists of multiple `noether.core.models.Model` sub-modules, each potentially with its own:
- Optimizer and learning rate
  - Learning rate schedule
- Weight initialization strategy
- Frozen/trainable status

**Example:** [CompositeTransformer](./model/composite_transformer.py) demonstrates a Transformer model with two sub-modules, each with independent configurations.

**Configuration files:**
- **Schema**: [Composite Transformer config](./schemas/models/composite_transformer_config.py#11) defines the overall structure
- **Sub-module schemas**: [Sub-modules](./configs/model/composite_transformer.yaml#22) each have their own config schemas
- **YAML example**: [composite_transformer.yaml](./configs/model/composite_transformer.yaml) shows how to configure different optimizers and learning rates per sub-module

**Example configuration snippet (note that classes do not exist):**
```yaml
kind: tutorial.model.composite_transformer.CompositeTransformer
name: composite_transformer
encoder:
  kind: tutorial.model.transformer.TransformerEncoder
  hidden_dim: 192
  optimizer_config:
    kind: torch.optim.AdamW
    lr: 1e-3
decoder:
  kind: tutorial.model.transformer.TransformerDecoder
  hidden_dim: 192
  optimizer_config:
    kind: torch.optim.Lion
    lr: 5e-4
```

### `Noether` model zoo

The `Noether Framework` includes base implementations for several state-of-the-art models in [noether.modeling.models](../src/noether/modeling/models/):

| Model | Paper | Tutorial Implementation | Notes |
|-------|-------|------------------------|-------|
| **AB-UPT** | [arXiv:2502.09692](https://arxiv.org/abs/2502.09692) | [tutorial/model/ab_upt.py](./model/ab_upt.py) | Wrapper around base implementation |
| **Transformer** | - | [tutorial/model/transformer.py](./model/transformer.py) | Wrapper around base implementation and adding RoPE |
| **Transolver** | [arXiv:2402.02366](https://arxiv.org/abs/2402.02366) | [tutorial/model/transolver.py](./model/transolver.py) | Wrapper around base implementation |
| **Transolver++** | [arXiv:2502.02414](https://arxiv.org/abs/2502.02414) | Schema only: [transolver.py](../src/noether/core/schemas/models/transolver.py#21) | Extension of Transolver with different attention class |
| **UPT** | [arXiv:2402.12365](https://arxiv.org/abs/2402.12365) | [tutorial/model/upt.py](./model/upt.py) | Extended `forward` method for tutorial compatibility |

**Implementation approaches:**
- **Simple wrappers**: AB-UPT, Transformer, and Transolver use the base implementations directly
- **Custom extensions**: UPT uses individual sub-modules from the base implementation with a modified `forward` method to adapt to tutorial-specific requirements

## Callbacks ([`callbacks/`](./callbacks/))

A **callback** is an object that can perform actions at various stages of the training loop, such as at the beginning or end of training, an epoch, or an update step.
Callbacks are the most complex objects in the `Noether Framework`. 
For a full understanding of callback implementation and utilities, refer to the documentation: [TO ADD].

### Overview

The [SurfaceVolumeEvaluationMetricsCallback](./callbacks/surface_volume_evaluation_metrics.py) is a specific callback that runs the current model on a separate validation or test set, computes error metrics, and logs them.
This class inherits from **`PeriodicIteratorCallback`**, meaning its main logic is executed at regular intervals and iterates over a dataset.
In this tutorial, we focus only on **`PeriodicIteratorCallback`**.
However, you can also implement a **`PeriodicCallback`**, which does not iterate over a dataset but can be used, for example, to store an exponential moving average (EMA) of the model weights.

**Callback access to training components:**

Callbacks have access to the following (among others):
- **The Trainer (`self.trainer`)**: Provides access to trainer properties
- **The Model (`self.model`)**: The currently trained model
- **The Data Container (`self.data_container`)**: Object containing all datasets, allowing normalizers to be accessed for denormalization

### Implementing PeriodicIteratorCallback

Callbacks that inherit from **`PeriodicIteratorCallback`** must implement two methods:

1. **`_forward(self, batch: dict[str, torch.Tensor], **_) -> dict[str, torch.Tensor]`**: Receives a batch from the dataset as input and computes metrics (or tensors) that are returned.
2. **`_process_results(self, results: dict[str, torch.Tensor], **_) -> None`**: All computed metrics/tensors from the **`_forward`** method are aggregated into a dictionary and processed by this method.

For example, the **`_process_results`** method can use **`self.writer`** to log metrics to Weights & Biases.

### Configuring callbacks

In [**`tutorial/configs/trainer/shapenet_trainer.yaml`**](./configs/trainer/shapenet_trainer.yaml), we define the list of callbacks to use for the trainer class (for ShapeNet-Car).
Below are three callback configurations:

```yaml
- kind: noether.core.callbacks.BestCheckpointCallback
  every_n_epochs: 1
  metric_key: loss/test/total
  name: BestCheckpointCallback
# test loss
- kind: tutorial.callbacks.SurfaceVolumeEvaluationMetricsCallback
  batch_size: 1
  every_n_epochs: 1
  dataset_key: test
  name: SurfaceVolumeEvaluationMetricsCallback
  forward_properties: ${model.forward_properties}
- kind: tutorial.callbacks.SurfaceVolumeEvaluationMetricsCallback
  batch_size: 1
  every_n_epochs: ${trainer.max_epochs}
  dataset_key: test_repeat
  name: SurfaceVolumeEvaluationMetricsCallback
  forward_properties: ${model.forward_properties}
```

**Periodic callback triggers:**

To define how often a periodic callback (i.e., **`_periodic_callback`**) should be triggered, set one of the following arguments in your configuration:

- **`every_n_epochs`**: Triggers the callback every N epochs
- **`every_n_updates`**: Triggers the callback every N model update steps
- **`every_n_samples`**: Triggers the callback after every N samples have been processed

You cannot define multiple of these arguments. In addition to the interval, you can also define the **`batch_size`**, which is usually set to `1` to compute metrics per sample.

**Required callback parameters:**

For all periodic callbacks, you must define:
- **`dataset_key`**: Indicates which dataset (configured earlier) should be used to run the callback
- **`name`**: Must match a name in the callback schemas so that the correct schema can be used for data validation

### Schema validation for callbacks

In [**`tutorial.schemas.trainers.AutomotiveAerodynamicsCfdTrainerConfig`**](./schemas/trainers/automotive_aerodynamics_trainer_config.py), we define the following for callback validation:

```python

from noether.core.schemas.callbacks import CallbacksConfig
from tutorial.schemas.callbacks import TutorialCallbacksConfig

AllCallbacks = Union[
    TutorialCallbacksConfig, CallbacksConfig
]  # custom callbacks need to be added here to one union type with the base noether CallbacksConfig

class AutomotiveAerodynamicsCfdTrainerConfig(BaseTrainerConfig):
  ...
  callbacks: list[AllCallbacks] | None = Field(
        ...,
    )  
```

You need to define a Union of your custom-implemented callbacks and the default callbacks implemented in the `Noether Framework` to ensure all callbacks have proper schema validation.

### _forward implementation

The `_forward` method of the `SurfaceVolumeEvaluationMetricsCallback` method simply looks like:
```python
def _forward(self, batch: dict[str, torch.Tensor], **_) -> dict[str, torch.Tensor]:
        """
        Execute forward pass and compute metrics.

        Args:
            batch: Input batch dictionary
            **_: Additional unused arguments

        Returns:
            Dictionary mapping metric names to computed values
        """
        model_outputs = self._run_model_inference(batch)

        metrics = {}
        for mode in self.evaluation_modes:
            metrics.update(self._compute_mode_metrics(batch, model_outputs, mode))

        return metrics

```

First it computes the module outputs, and next it adds the desired metrics to an output dictionary.
All the substeps are implemented by individual methods in the callback itself.
Please have a look at the [implementation](./callbacks/surface_volume_evaluation_metrics.py)

### Denormalization for metrics

Metrics are usually computed on unnormalized data. 
To denormalize the normalization steps executed by the dataset, we retrieve the data normalizers via the **`DataContainer`**. 
In the **`__init__`** method of the callback we implement, we use the available **`self.data_container`** to get the correct dataset used for this callback and retrieve the normalizers to denormalize the data for metric computation:

```python
self.dataset_key = callback_config.dataset_key
self.dataset_normalizers = self.data_container.get_dataset(self.dataset_key).normalizers
```

To denormalize **`surface_pressure`**, for example, you can use:

```python
normalizer = self.dataset_normalizers['surface_pressure']
denormalized_predictions = normalizer.inverse(predictions.cpu())
denormalized_targets = normalizer.inverse(targets.cpu())
```

### Computed metrics

For each output in the **`SurfaceVolumeEvaluationMetricsCallback`**, we calculate the following metrics:

1. **Mean Squared Error (MSE)**: The average of the squared differences between the prediction and the target
2. **Mean Absolute Error (MAE)**: The average of the absolute differences between the prediction and the target
3. **Relative L2 Error**: The Euclidean norm of the error vector divided by the norm of the target vector, measuring the error relative to the magnitude of the ground truth

### Final evaluation with repeated testing

At the end of training, we want to run the model one more time on the test set, looping 10 times over that set to reduce variance due to the point sampling.
Earlier, we configured the **`test_repeat`** dataset in [**`shapenet_dataset.yaml`**](./configs/datasets/shapenet_dataset.yaml), which uses the **`RepeatWrapper`** to loop over the dataset with 10 repetitions. 
We can now use **`test_repeat`** with this custom dataset implementation for the final callback.
Moreover, we set **`every_n_epochs: ${trainer.max_epochs}`** to ensure that this callback is only executed at the very end. 
Each metric is logged with the corresponding **`dataset_key`** to Weights & Biases.

For the `CAEML` dataset, we also implemented chunked inference, where we loop over the entire surface and volume mesh in chunks to do inference on the full mesh.
To enable this we set `chunked_inference: true`, and we configured a dataset `chunked_test` which has a multi-stage pipeline that returns all points in the surface/volume mesh.

## Running the experiments

### Running SLURM jobs

To run all the models for ShapeNet-Car, simply execute:

```bash
sbatch train_shapenet.job
```

The same applies to `train_ahmedml.job` and `train_drivaerml.job`, which can be found in the [**`jobs/`**](./jobs/) directory.
We also provide the config files to run the experiments for [DrivAerNet++](https://arxiv.org/abs/2406.09624) ([train_drivaernet.yaml](./configs/train_drivaernet.yaml)) and the [Emmi-Wing](http://arxiv.org/html/2511.21474v1) ([train_wing.yaml](./configs/train_wing.yaml)), however, those experiments are not part of this tutorial. 


> [!WARNING]
> This assumes you have access to a SLURM-based system. If not, please review the job files to see the commands used to run the experiments.

**Job arrays:**

In the [**`jobs/experiments/`**](./jobs/experiments/) folder, we define job arrays (i.e., arrays with different experiments/jobs) for all the experiments we want to run.
You can add extra rows with different seeds or experiment variants to these `*.txt` files as needed.

The flag `#SBATCH --array=...` defines how to run the job array:

- **`#SBATCH --array=1-10`**: Runs rows 1 to 10 from `./jobs/experiments/shapenet_experiments.txt`
- **`#SBATCH --array=1,5,9`**: Runs rows 1, 5, and 9
- **`#SBATCH --array=1-10%5`**: Runs rows 1 to 10, but with a maximum of 5 jobs running simultaneously. When one of the 5 jobs finishes, the next job in the array (e.g., row 6) will start. This is especially useful for large job arrays when you don't want to occupy the entire cluster.

### Running a single experiment

To run a single experiment, execute the following command:

```bash
uv run noether-train --hp {user path to tutorial}/configs/train_shapenet.yaml +experiment/shapenet=transformer tracker=disabled +seed=1
```

> [!IMPORTANT]
> Please set the `dataset_root` in either the config files or via the command line override

### Running multi-GPU experiments

When running outside of SLURM, use `uv run noether-train` as shown above. This will spawn one process for every GPU that is available on the system and visible via `CUDA_VISIBLE_DEVICES`. 
You can also configure the devices by adding `devices="0,1,2,4"`, for example, to the root config. 

> [!IMPORTANT]
> If you train on more than 1 GPU, ensure that `effective_batch_size` is at least equal to the number of GPUs used.
Multi-node training is currently not supported.

Example of a multi-gpu SLURM job:

```bash
srun --nodes=1 --partition=compute --gpus-per-node=2 --mem=64GB --ntasks-per-node=2 --kill-on-bad-exit=1 --cpus-per-task=28 uv run noether-train --hp tutorial/configs/train_shapenet.yaml +experiment/shapenet=transformer tracker=disabled trainer.effective_batch_size=2
```

### Running inference

To run evaluation callbacks on trained models, use the `noether-eval` CLI tool.

For detailed instructions on running inference with trained models, refer to the documentation:
https://noether-docs.emmi.ai/guides/inference/how_to_run_evaluation_on_trained_models.html

### Resuming training after interruption

To resume training after an error or interruption, simply add `resume_run_id: <RUN_ID>` to the training configuration (either in the YAML file or via the CLI). Training will continue from the last saved epoch checkpoint.

**Example:**
```bash
uv run noether-train --hp tutorial/configs/train_shapenet.yaml +experiment/shapenet=transformer resume_run_id=<run_id>
```

### Initializing model weights

To initialize a model with weights from a previous training run, add an initializer configuration to the model config:

```yaml
model:
  # ... model configuration
  initializers:
    - kind: noether.core.initializers.PreviousRunInitializer
      run_id: <run_id>
      model_name: ab_upt
      checkpoint: latest  # Options: 'latest', 'best', or specific epoch number
      # model_info: ema=0.9999  # Optional: for EMA weights or specific checkpoint variants
```

**Required parameters:**
- **`run_id`**: The run identifier from the previous training run
- **`model_name`**: The name of the model to load weights from
- **`checkpoint`**: Which checkpoint to use (`latest`, `best`, or a specific epoch number)

**Optional parameters:**
- **`model_info`**: Additional checkpoint metadata (e.g., `ema=0.9999` for exponential moving average weights, or specific loss metric identifiers for best checkpoints). Leave empty for standard checkpoints. 

### WandB tracker

We implemented a Weights and Biases (WandB) tracker to log during training and evaluation (also have a look at [./configs/tracker](./configs/tracker/)).

```yaml
kind: noether.core.trackers.WandBTracker
entity: <WandB entity> 
project: <WandB project> 
```

Simply add your own WandB entity and project to start logging. 

## Extra utilities and tips

- **Output path**: The output path is undefined by default and must be configured. In this tutorial, we set it to `./outputs`. The `Noether Framework` will use the generated `run_id` to store the checkpoints for each training run in subfolders.
- **Physics features**: You can set `physics_features` to `true` for the multi-stage `AeroMultistagePipeline`. This only works for ShapeNet-Car and will add the SDF and normal vectors to the coordinate inputs. However, we never properly utilized these features in our experiments, and they are not implemented for other datasets. Therefore, this code is not fully polished or optimized.
- **Code snapshots**: By default, a snapshot of the codebase is stored as part of the checkpoints for reproducibility.
- **Batch size considerations**: Almost all experiments we ran for the AB-UPT paper use a batch size of 1. However, the data loading pipeline is implemented to work with batches larger than 1 (including with physics features). Note that we never thoroughly validated these results or checked for potential training/data loading instabilities with larger batch sizes.
- **Effective batch size and gradient accumulation**: The `effective_batch_size` parameter defines the total number of samples processed before performing an optimizer step (also known as the "global batch size"). In multi-GPU setups, the local batch size per device is calculated as `effective_batch_size / number of GPUs`. When gradient accumulation is enabled, the batch size is further divided by the number of accumulation steps. To enable gradient accumulation, set the `max_batch_size` parameter. For example, with `max_batch_size=2` and `effective_batch_size=8`, the framework will perform 4 gradient accumulation steps before updating the model weights. 
