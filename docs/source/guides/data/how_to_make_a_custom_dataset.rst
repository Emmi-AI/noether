How to Implement a Custom Dataset
=================================

Below we provide a minimal (dummy code) example of how to create a custom dataset by extending the base :py:class:`~noether.data.dataset.Dataset` class.
Every single tensor that belongs to a 'data sample' must have its own ``getitem_*`` method, with a unique suffix.
By default, all ``getitem_*`` will be called when fetching a data sample, unless specified otherwise in the configuration file (by configuring ``excluded_properties``).
To apply data normalization, the ``@with_normalizers`` decorator must be used on each ``getitem_*`` method.
The key provided to the decorator must match the key of the configured normalizer.

.. testcode::

    from noether.data import Dataset, with_normalizers
    from noether.core.schemas.dataset import DatasetBaseConfig
    import torch
    import os

    class MyCustomDatasetConfig(DatasetBaseConfig):
        kind: str = "path.to.MyCustomDataset"
        # Add any custom configuration fields here
        data_paths: dict[int, str]

    class MyCustomDataset(Dataset):
        def __init__(self, config: MyCustomDatasetConfig):
            super().__init__(config)
            self.data_paths = config.data_paths
            self.root = config.root

        def __len__(self):
            # Return the length of your dataset
            return len(self.data_paths)

        @with_normalizers("tensor_x")
        def getitem_tensor_x(self, idx: int) -> torch.Tensor:
            # Load and return the data sample and its corresponding label as tensors
            return torch.load(os.path.join(self.root, self.data_paths[idx]), weights_only=True)

        @with_normalizers("tensor_y")
        def getitem_tensor_y(self, idx: int) -> torch.Tensor:
            # Load and return the data sample and its corresponding label as tensors
            return torch.load(os.path.join(self.root, self.data_paths[idx]), weights_only=True)

.. testcode::
   :hide:

   _cfg = MyCustomDatasetConfig(kind="path.to.MyCustomDataset", data_paths={0: "a.pt"})

.. code-block:: yaml

    datasets:
        custom_dataset:
            kind: path.to.MyCustomDataset
            root: /path/to/data
            data_paths:
                0: sample_0.pt
                1: sample_1.pt
                2: sample_2.pt
                # Add more data paths as needed
            excluded_properties: []  # Optionally exclude certain getitem_* methods


.. code-block:: yaml

    tensor_x:
        - kind: noether.data.preprocessors.normalizers.MeanStdNormalization
          mean: 0.0
          std: 1.0
    tensor_y:
        - kind: noether.data.preprocessors.normalizers.MeanStdNormalization
          mean: 1.0
          std: 2.0


Using ``pre_getitem`` / ``post_getitem`` Hooks
-----------------------------------------------

When multiple properties are stored inside a single file (e.g. an HDF5 container), opening the file
separately in every ``getitem_*`` method is wasteful.  Override
:py:meth:`~noether.data.Dataset.pre_getitem` to load shared data once, and optionally
:py:meth:`~noether.data.Dataset.post_getitem` for cleanup.  The dictionary returned by
``pre_getitem`` is forwarded as keyword arguments to every ``getitem_*`` method that declares
matching parameters.

.. testcode::

    from noether.data import Dataset
    from noether.core.schemas.dataset import DatasetBaseConfig

    class HDF5DatasetConfig(DatasetBaseConfig):
        kind: str = "path.to.HDF5Dataset"
        file_paths: dict[int, str] = {}

    class HDF5Dataset(Dataset):
        def __init__(self, config: HDF5DatasetConfig):
            super().__init__(config)
            self.file_paths = config.file_paths

        def __len__(self):
            return len(self.file_paths)

        def pre_getitem(self, idx: int) -> dict:
            # Open the file once per sample
            return {"h5file": h5py.File(self.file_paths[idx], "r")}

        def post_getitem(self, idx: int, pre: dict | None) -> None:
            # Guaranteed to run, even if a getitem_* method raises
            if pre is not None:
                pre["h5file"].close()

        def getitem_temperature(self, idx: int, *, h5file=None, **kwargs):
            return torch.tensor(h5file["temperature"][:])

        def getitem_pressure(self, idx: int, *, h5file=None, **kwargs):
            return torch.tensor(h5file["pressure"][:])

.. testcode::
   :hide:

   _cfg = HDF5DatasetConfig(kind="path.to.HDF5Dataset")

``getitem_*`` methods that only accept ``idx`` (like the basic example above) continue to work
unchanged -- they simply won't receive the extra keyword arguments.


Let's say we run a training pipeline with the above dataset configuration and a batch size of 4.
Then the output will be a list of 4 dictionaries (a batch), where each dictionary has the following structure:

.. code-block:: python

    [
        {
            "tensor_x": <tensor_x_sample_0>,
            "tensor_y": <tensor_y_sample_0>,
        },
        # ... (3 more samples)
    ]