How to Build a Custom Dataset
==============================

Below we provide a minimal (dummy code) example of how to create a custom dataset by extending the base ``Dataset`` class.
Every single tensor that belongs to a 'data sample' must have its own ``getitem_*`` method, with a unique suffix.
By default, all ``getitem_*`` will be called when fetching a data sample, unless specified otherwise in the configuration file (by configuring ``excluded_properties``).
To apply data normalization, the ``@with_normalizers`` decorator must be used on each ``getitem_*`` method.
The key provided to the decorator must match the key of the configured normalizer.

.. code-block:: python

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