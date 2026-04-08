Datasets
========

Object instantiation
--------------------

The objects instantiated in the Noether Framework via configs use a **factory pattern**.
The config of each object contains a ``kind`` field --- the class path of the class to instantiate.
The remaining variables are passed to the constructor via the config object created after Pydantic
schema evaluation. For example, ``kind: noether.modeling.models.AeroTransformer`` indicates which
model class to instantiate.


The Dataset
-----------

The ``Dataset`` class serves as the bridge between raw (or preprocessed) data stored on disk
and the **multi-stage pipeline** that transforms individual samples into batches for model
training (discussed in :doc:`pipeline`). It defines how to load and access individual data
tensors for each sample.

**The Dataset enables:**

- Loading individual data samples from disk
- Providing tensor-level data access through modular methods
- Applying per-tensor normalization and transformations
- Supporting flexible data loading for different model requirements

This walkthrough uses the pre-implemented
:py:class:`~noether.data.datasets.cfd.ShapeNetCarDataset`
from the Noether package.

**Dataset class hierarchy:**

.. code-block:: text

   torch.utils.data.Dataset (PyTorch base)
       └── noether.data.Dataset (Noether base with getitem_* pattern)
             └── noether.data.datasets.cfd.AeroDataset (CFD aerodynamics API)
                   └── ShapeNetCarDataset (ShapeNet-Car implementation)

The ``AeroDataset`` provides a general API for CFD aerodynamics datasets (AhmedML, DrivAerML,
DrivAerNet++, ShapeNet-Car, etc.), ensuring consistent interfaces across different aerodynamics datasets.

For a concise guide on building your own dataset, see
:doc:`/guides/data/how_to_make_a_custom_dataset`.


The ``getitem_*`` pattern: Modular data loading
------------------------------------------------

Traditional PyTorch datasets use a single ``__getitem__`` method to load all data for a sample.
This approach has several limitations:

- Becomes complex when different models need different inputs from the same dataset
- Difficult to selectively load subsets of data
- Hard to maintain when adding new data fields
- Forces loading unused data for some experiments

The Noether Framework uses a **modular** ``getitem_*`` **pattern** where each data tensor has
its own dedicated loading method. This enables:

- **Modularity**: Each method loads one specific tensor
- **Flexibility**: Selectively load only required tensors via configuration
- **Maintainability**: Easy to add new data fields without modifying existing code
- **Clarity**: Self-documenting through method names (e.g., ``getitem_surface_pressure``)

**Example implementation:**

.. code-block:: python

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

**Design pattern:**

- **Helper methods** (e.g., ``_load``) keep code DRY and handle common operations
- **Descriptive names** make it clear what each method loads
- **Consistent signature**: All ``getitem_*`` methods take ``idx`` and return a tensor
- **Tensor-level operations**: Shape transformations (e.g., ``unsqueeze``) applied immediately


ShapeNet-Car dataset structure
------------------------------

The ShapeNet-Car dataset contains CFD simulation data for 889 car geometries, with each data
point consisting of preprocessed PyTorch tensors stored on disk.

.. note::

   To download and preprocess the data, see the
   `ShapeNet-Car dataset README <https://github.com/Emmi-AI/noether/blob/main/src/noether/data/datasets/cfd/shapenet_car/README.MD>`_.

**Available data tensors:**

Each simulation provides the following fields through corresponding ``getitem_*`` methods:

.. list-table::
   :widths: 20 25 15 40
   :header-rows: 1

   * - Tensor
     - Method
     - Shape
     - Description
   * - **Surface Position**
     - ``getitem_surface_position``
     - ``(N_surf, 3)``
     - 3D coordinates of surface mesh points
   * - **Surface Pressure**
     - ``getitem_surface_pressure``
     - ``(N_surf, 1)``
     - Pressure values at surface points
   * - **Surface Normals**
     - ``getitem_surface_normals``
     - ``(N_surf, 3)``
     - Normal vectors at surface points
   * - **Volume Position**
     - ``getitem_volume_position``
     - ``(N_vol, 3)``
     - 3D coordinates of volume mesh points
   * - **Volume Velocity**
     - ``getitem_volume_velocity``
     - ``(N_vol, 3)``
     - Velocity vectors at volume points
   * - **Volume Normals**
     - ``getitem_volume_normals``
     - ``(N_vol, 3)``
     - Normal vectors (pointing to nearest surface)
   * - **Volume SDF**
     - ``getitem_volume_sdf``
     - ``(N_vol, 1)``
     - Signed Distance Field to nearest surface


Dataset configuration
---------------------

Datasets in Noether are instantiated by the ``DatasetFactory``, which uses configuration files
to create dataset instances with appropriate settings.

**Basic dataset configuration structure:**

The :source:`configs/datasets/shapenet_dataset.yaml <../../../../recipes/aero_cfd/configs/datasets/shapenet_dataset.yaml>` file defines dataset configurations for
different splits:

.. literalinclude:: ../../../../recipes/aero_cfd/configs/datasets/shapenet_dataset.yaml
   :language: yaml

**Configuration parameters:**

- ``root``: Path to the dataset directory on disk
- ``kind``: Full class path to the dataset class (e.g., ``noether.data.datasets.cfd.ShapeNetCarDataset``)
- ``split``: Data split identifier (``train``, ``test``, ``val``, etc.) used by the dataset to select appropriate samples
- ``pipeline``: Reference to the multi-stage pipeline configuration
- ``dataset_normalizers``: Reference to tensor normalization configurations
- ``excluded_properties``: List of ``getitem_*`` methods to skip during data loading

The ``test_repeat`` section demonstrates multiple dataset configurations for different
evaluation scenarios.

**Dataset wrappers:**

The ``RepeatWrapper`` loops over the dataset multiple times (10x in this example) to reduce
variance during evaluation. Other useful wrappers include:

- ``SubsetWrapper``: Select specific indices from the dataset
- ``ShuffleWrapper``: Randomize sample order

This flexibility allows you to:

- Use different pipelines for train vs. test datasets
- Create multiple evaluation sets with different sampling strategies
- Apply different normalizations to different splits


Selective data loading with ``excluded_properties``
---------------------------------------------------

By default, all ``getitem_*`` methods are called when loading a sample. However, different
models often require different input tensors. The ``excluded_properties`` configuration allows
selective loading:

.. code-block:: yaml

   # Example: Exclude normal vectors for a model that doesn't use them
   excluded_properties:
     - surface_normals
     - volume_normals

A point-based Transformer might only need positions, surface pressure, and volume velocity:

.. code-block:: yaml

   # Load only essential tensors
   excluded_properties:
     - surface_normals
     - volume_normals
     - volume_sdf

A more complex model can use all available features:

.. code-block:: yaml

   # Load everything
   excluded_properties: []

This pattern enables using the same dataset class for different models without modifying code.


Essential dataset methods
-------------------------

Beyond the ``getitem_*`` methods, dataset classes implement standard PyTorch dataset methods:

**__len__ method:**

Defines the total number of samples for one epoch:

.. code-block:: python

   def __len__(self) -> int:
       """Returns the total size of the dataset."""
       return len(self.uris)

This calculation accounts for dataset repetitions, useful for oversampling small datasets
during training.

Most other methods follow standard PyTorch ``Dataset`` patterns. If you're unfamiliar with
PyTorch datasets, review the
`official PyTorch dataset tutorial <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_.


Tensor normalization with decorators
-------------------------------------

In the Noether Framework, most of the normalization happens at the **tensor level** immediately
after loading, using a decorator pattern for clean, declarative code.

**The @with_normalizers decorator:**

Apply normalization to any ``getitem_*`` method by adding a decorator:

.. code-block:: python

   @with_normalizers
   def getitem_surface_position(self, idx: int) -> torch.Tensor:
       """Retrieves surface positions (num_surface_points, 3)"""
       return self._load(idx=idx, filename=self.filemap.surface_position)

By default, the decorator infers the normalizer key from the method name (stripping the
``getitem_`` prefix). You can pass an explicit key when the normalizer name differs from
the method name --- for example, ``@with_normalizers("volume_sdf")`` on the
``getitem_surface_sdf`` method to reuse the volume SDF normalizer.

**How it works:**

#. The decorator identifies which normalizer(s) to apply using the key (derived from the method name, or explicitly provided)
#. Looks up the normalizer configuration in the dataset's ``dataset_normalizers`` config
#. Applies the normalization transformation to the loaded tensor
#. Returns the normalized tensor

**Configuring normalizers:**

All normalizers are defined in :py:mod:`noether.data.preprocessors.normalizers`. The
:py:class:`~noether.data.preprocessors.normalizers.FieldNormalizer` is the primary normalizer,
which supports different strategies (``mean_std``, ``position``, etc.):

.. literalinclude:: ../../../../recipes/aero_cfd/configs/dataset_normalizers/shapenet_dataset_normalizers.yaml
   :language: yaml

Each key maps to a normalizer configuration. The ``strategy`` field selects the normalization
method --- for example, ``mean_std`` for standard mean/std normalization, or ``position`` for
coordinate normalization with min/max scaling. All normalizers must be invertible so that data
can be denormalized for evaluation.


Computing dataset statistics
-----------------------------

To use normalizers like ``MeanStdNormalization``, you need to compute statistics from your
training data.

**Step 1: Compute statistics**

Run the statistics calculation tool:

.. code-block:: bash

   noether-dataset-stats \
     --dataset_kind=noether.data.datasets.cfd.ShapeNetCarDataset \
     --root=/path/to/shapenet_car/ \
     --split=train \
     --exclude_attributes=volume_velocity,volume_pressure,volume_vorticity,surface_normals,surface_friction

**Parameters explained:**

- ``--dataset_kind``: Full class path to your dataset
- ``--root``: Path to dataset directory
- ``--split``: Which split to compute statistics from (typically ``train``)
- ``--exclude_attributes``: Properties to skip (either unavailable or not used)

.. note::

   We exclude certain properties because they're not available in ShapeNet-Car, even though
   the general ``AeroDataset`` interface defines ``getitem_*`` methods for them.

The statistics need to be manually added to a YAML file in ``configs/dataset_statistics/``.


Noether dataset zoo
-------------------

The Noether Framework includes pre-implemented datasets for CFD aerodynamics. For a complete
listing, see :doc:`/noether/dataset_zoo`.

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Dataset
     - Class Path
     - Data Processing README
   * - **ShapeNet-Car**
     - :py:class:`~noether.data.datasets.cfd.ShapeNetCarDataset`
     - `ShapeNet-Car README <https://github.com/Emmi-AI/noether/blob/main/src/noether/data/datasets/cfd/shapenet_car/README.MD>`__
   * - **AhmedML**
     - :py:class:`~noether.data.datasets.cfd.AhmedMLDataset`
     - `AhmedML README <https://github.com/Emmi-AI/noether/blob/main/src/noether/data/datasets/cfd/caeml/README.MD>`__
   * - **DrivAerML**
     - :py:class:`~noether.data.datasets.cfd.DrivAerMLDataset`
     - `DrivAerML README <https://github.com/Emmi-AI/noether/blob/main/src/noether/data/datasets/cfd/caeml/README.MD>`__
   * - **DrivAerNet++**
     - :py:class:`~noether.data.datasets.cfd.DrivAerNetDataset`
     - `DrivAerNet++ README <https://github.com/Emmi-AI/noether/blob/main/src/noether/data/datasets/cfd/drivaernet/README.MD>`__
   * - **Wing Dataset**
     - :py:class:`~noether.data.datasets.cfd.EmmiWingDataset`
     - `Wing README <https://github.com/Emmi-AI/noether/blob/main/src/noether/data/datasets/cfd/emmi_wing/README.MD>`__

All datasets share the ``AeroDataset`` interface, ensuring consistent access patterns and easy
switching between datasets.


Creating custom datasets
------------------------

To implement a custom dataset:

#. Inherit from ``noether.data.Dataset`` (or ``noether.data.datasets.cfd.AeroDataset``)
#. Implement required ``getitem_*`` methods for your data fields
#. Override ``__init__`` to discover and filter your data samples
#. Add ``@with_normalizers`` decorators where normalization is needed
#. Create a corresponding Pydantic schema in your ``schemas/datasets/`` directory
#. Configure the normalizers

See the scaffold template (``src/noether/scaffold/template_files/datasets/``) for a minimal
dataset implementation example, or run ``noether-init`` to generate a ready-to-use project
(see :doc:`/tutorials/scaffolding_a_new_project`). For a step-by-step guide, see
:doc:`/guides/data/how_to_make_a_custom_dataset`.
